import json
import signal
import sys
import time

import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

import config
import metrics
import serving
from fdd import compute_fdd


def reshape_baseline(raw: dict) -> dict:
    """Feast feature dict -> numpy baseline ready for compute_fdd."""
    cov_shape = [int(x) for x in raw["cov_shape"]]
    pcc_shape = [int(x) for x in raw["pca_components_shape"]]
    return {
        "mean": np.asarray(raw["mean"], dtype=np.float64),
        "cov": np.asarray(raw["cov"], dtype=np.float64).reshape(cov_shape),
        "pca_components": np.asarray(raw["pca_components"], dtype=np.float64).reshape(pcc_shape),
        "pca_mean": np.asarray(raw["pca_mean"], dtype=np.float64),
        "threshold": float(raw["threshold"]),
        "reference_set_id": raw["reference_set_id"],
        "baseline_id": raw["_active_baseline_id"],
    }


def run() -> int:
    metrics.serve(config.METRICS_PORT)
    print(f"[detector] /metrics on :{config.METRICS_PORT}", flush=True)

    consumer = Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP,
        "group.id": config.GROUP_ID,
        # commit only after a full window
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "session.timeout.ms": 45000,
        "socket.keepalive.enable": True,
    })
    consumer.subscribe([config.INPUT_TOPIC])
    producer = Producer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP,
        "socket.keepalive.enable": True,
        "message.timeout.ms": 120000,
    })

    state = {"baseline": None, "paused": False, "backoff": config.BACKOFF_START,
             "buffer": [], "window_index": 0, "running": True}

    def shutdown(*_):
        state["running"] = False
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    def pause():
        a = consumer.assignment()
        if a:
            consumer.pause(a)
        state["paused"] = True

    def resume():
        a = consumer.assignment()
        if a:
            consumer.resume(a)
        state["paused"] = False
        state["backoff"] = config.BACKOFF_START

    def backoff_sleep():
        time.sleep(state["backoff"])
        state["backoff"] = min(state["backoff"] * 2, config.BACKOFF_MAX)

    def check_ready() -> bool:
        """Readiness gate. Sets the reason metric and (re)loads the baseline on
        a served/changed pointer. Returns True iff served."""
        reason, raw = serving.baseline_readiness(
            config.FEAST_URL, config.MODEL_VERSION, config.HTTP_TIMEOUT)
        metrics.set_readiness(reason)
        if reason != "served":
            return False
        b = state["baseline"]
        if b is None or b["baseline_id"] != raw["_active_baseline_id"]:
            state["baseline"] = reshape_baseline(raw)
            metrics.drift_threshold.labels(
                config.MODEL_VERSION, state["baseline"]["reference_set_id"]
            ).set(state["baseline"]["threshold"])
            print(f"[detector] baseline loaded: {state['baseline']['baseline_id']} "
                  f"(threshold={state['baseline']['threshold']:.4f})", flush=True)
        return True

    def score_window():
        b = state["baseline"]
        E_w = np.vstack(state["buffer"][:config.WINDOW_SIZE])
        fdd = compute_fdd(E_w, b)
        drift = fdd > b["threshold"]
        metrics.drift_score.record(fdd, config.MODEL_VERSION, b["reference_set_id"])
        metrics.drift_windows_processed_total.inc()
        if drift:
            metrics.drift_detected_total.inc()
        # best-effort: a broker hiccup must not crash the detector
        try:
            producer.produce(config.OUTPUT_TOPIC, key=config.MODEL_VERSION, value=json.dumps({
                "model_version": config.MODEL_VERSION,
                "baseline_id": b["baseline_id"],
                "reference_set_id": b["reference_set_id"],
                "window_index": state["window_index"],
                "fdd": fdd,
                "threshold": b["threshold"],
                "drift": bool(drift),
                "n": config.WINDOW_SIZE,
                "ts": time.time(),
            }))
            producer.poll(0)
        except (KafkaException, BufferError) as e:
            print(f"[detector] WARN drift-signals emit failed: {e}", flush=True)
        # commit after the full window; a transient timeout retries next window
        try:
            consumer.commit(asynchronous=False)
        except KafkaException as e:
            print(f"[detector] WARN commit failed (retry next window): {e}", flush=True)
        state["buffer"] = state["buffer"][config.WINDOW_SIZE:]
        state["window_index"] += 1
        print(f"[detector] window {state['window_index']-1}: fdd={fdd:.4f} "
              f"thr={b['threshold']:.4f} drift={drift}", flush=True)

    try:
        while state["running"]:
            # gate only when we don't hold a baseline; per-window re-check is below
            if state["baseline"] is None or state["paused"]:
                if not check_ready():
                    pause()
                    backoff_sleep()
                    continue
                resume()

            msg = consumer.poll(config.POLL_TIMEOUT)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[detector] kafka error: {msg.error()}", flush=True)
                continue
            try:
                emb = np.asarray(json.loads(msg.value())["embedding"], dtype=np.float64)
            except (ValueError, KeyError, TypeError) as e:
                print(f"[detector] skip bad message: {e}", flush=True)
                continue
            state["buffer"].append(emb)

            if len(state["buffer"]) >= config.WINDOW_SIZE:
                # re-check right before scoring; never score against a stale baseline
                if not check_ready():
                    pause()
                    backoff_sleep()
                    continue
                score_window()
    finally:
        print("[detector] shutting down — flushing + closing", flush=True)
        producer.flush(5)
        consumer.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
