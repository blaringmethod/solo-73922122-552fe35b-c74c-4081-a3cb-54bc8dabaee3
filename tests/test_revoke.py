"""Revoke (undo) of the most recent successful harvest round.

A mistaken correction or withdrawal published from the source must be
removable before the source itself is fixed: revoking restores the catalogue
rows the round touched, the publish watermark and the last-successful-run
pointer, and marks the round revoked — atomically, idempotently, and never
while a harvest is running.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import wait_for_outcome

from app.main import create_app
from app.oai import OaiClient
from app.store import Store


def catalog_map(store) -> dict:
    return {
        r["identifier"]: r
        for r in store.catalog_page(page_size=500)["records"]
    }


async def harvest_ok(harvester, store) -> str:
    control = await harvester.start_run()
    assert await wait_for_outcome(control) == "succeeded"
    return control.run_id


# ---------------------------------------------------------------- store level


async def test_revoke_restores_catalogue_watermark_and_last_run(
    harvester, store, repo
):
    r1 = await harvest_ok(harvester, store)
    baseline_catalog = catalog_map(store)
    baseline = store.published_state()

    # The source publishes a mistaken correction, a withdrawal and an
    # erroneous new record; the next round publishes all of them.
    await asyncio.sleep(1.05)
    repo.touch_record("oai:demo:0001", title="误更正的标题")
    repo.withdraw("oai:demo:0002")
    repo.add_record(title="误新增的记录", identifier="oai:demo:bad1")
    r2 = await harvest_ok(harvester, store)

    after = catalog_map(store)
    assert after["oai:demo:0001"]["title"] == "误更正的标题"
    assert "oai:demo:0002" not in after
    assert "oai:demo:bad1" in after

    result = store.revoke_run(r2)
    assert result["revoked"] is True
    assert result["restored"] == 2  # 0001 correction + 0002 withdrawal
    assert result["removed"] == 1  # bad1 was added by the revoked round

    # Affected records are back to their pre-round state, every other row
    # is untouched, and the publish pointers moved back to round 1.
    assert catalog_map(store) == baseline_catalog
    state = store.published_state()
    assert state["watermark"] == baseline["watermark"]
    assert state["last_run_id"] == r1
    assert state["success_count"] == baseline["success_count"]

    # History marks the round revoked; the previous round stays successful.
    run2 = store.get_run(r2)
    assert run2["status"] == "succeeded"
    assert run2["revoked_at"]
    assert store.get_run(r1)["revoked_at"] is None
    # The revocable target is now round 1.
    assert store.revocable_run()["id"] == r1


async def test_revoke_first_full_harvest_returns_to_empty(harvester, store):
    r1 = await harvest_ok(harvester, store)
    assert store.published_state()["catalog_size"] > 0

    result = store.revoke_run(r1)
    assert result["revoked"] is True

    state = store.published_state()
    assert state["watermark"] is None
    assert state["last_run_id"] is None
    assert state["success_count"] == 0
    assert state["catalog_size"] == 0
    assert store.revocable_run() is None


async def test_revoke_refused_while_harvest_is_running(harvester, store, repo):
    r1 = await harvest_ok(harvester, store)

    await asyncio.sleep(1.05)
    repo.set_meta("slow_seconds", "5")
    control = await harvester.start_run()
    await asyncio.sleep(0.2)  # let the round actually start
    assert store.get_run(control.run_id)["status"] == "running"

    result = store.revoke_run(r1)
    assert result["revoked"] is False
    assert result["reason"] == "harvest_running"
    # Nothing changed.
    assert store.get_run(r1)["revoked_at"] is None
    assert store.published_state()["last_run_id"] == r1

    await harvester.cancel_run(control.run_id, "测试清理", "user")
    await wait_for_outcome(control)
    repo.set_meta("slow_seconds", "0")

    # Once no round is running, the same revoke goes through.
    assert store.revoke_run(r1)["revoked"] is True


async def test_repeated_revoke_same_id_never_touches_earlier_round(
    harvester, store, repo
):
    r1 = await harvest_ok(harvester, store)
    await asyncio.sleep(1.05)
    repo.touch_record("oai:demo:0003", title="第二轮的改动")
    r2 = await harvest_ok(harvester, store)

    assert store.revoke_run(r2)["revoked"] is True
    watermark_after_revoke = store.published_state()["watermark"]

    # A repeated request for the same id is a no-op: round 1 must stay
    # published and un-revoked.
    again = store.revoke_run(r2)
    assert again["revoked"] is False
    assert again["reason"] == "already_revoked"
    assert store.get_run(r1)["revoked_at"] is None
    assert store.published_state()["watermark"] == watermark_after_revoke
    assert store.published_state()["last_run_id"] == r1
    assert store.published_state()["success_count"] == 1

    # Revoking the (now latest) round 1 explicitly still works.
    assert store.revoke_run(r1)["revoked"] is True
    assert store.published_state()["watermark"] is None
    assert store.published_state()["success_count"] == 0


async def test_revoke_rejects_non_latest_and_non_successful_rounds(
    harvester, store, repo
):
    r1 = await harvest_ok(harvester, store)
    await asyncio.sleep(1.05)
    repo.touch_record("oai:demo:0004", title="新的更正")
    r2 = await harvest_ok(harvester, store)

    # Round 1 is not the latest successful round any more.
    result = store.revoke_run(r1)
    assert result["revoked"] is False
    assert result["reason"] == "not_latest"

    # A cancelled round cannot be revoked.
    cancelled = store.create_run()
    assert store.cancel_run(cancelled["id"], "测试取消")
    result = store.revoke_run(cancelled["id"])
    assert result["revoked"] is False
    assert result["reason"] == "not_succeeded"

    # Unknown id.
    assert store.revoke_run("no-such-run")["reason"] == "not_found"

    # State untouched by all the rejected attempts.
    assert store.published_state()["last_run_id"] == r2
    assert store.get_run(r2)["revoked_at"] is None


async def test_revoke_rejects_round_without_snapshot(harvester, store):
    """Rounds committed before revoke support have no retained pre-state."""
    r1 = await harvest_ok(harvester, store)
    # Simulate a legacy round: drop the retained pre-state.
    store._conn.execute("DELETE FROM run_pre_state WHERE run_id = ?", (r1,))
    result = store.revoke_run(r1)
    assert result["revoked"] is False
    assert result["reason"] == "no_snapshot"
    assert store.get_run(r1)["revoked_at"] is None


async def test_new_round_after_revoke_fetches_from_restored_watermark(
    harvester, store, repo
):
    r1 = await harvest_ok(harvester, store)
    watermark_r1 = store.published_state()["watermark"]

    await asyncio.sleep(1.05)
    repo.touch_record("oai:demo:0005", title="误更正")
    r2 = await harvest_ok(harvester, store)
    assert store.published_state()["watermark"] != watermark_r1

    assert store.revoke_run(r2)["revoked"] is True
    assert store.published_state()["watermark"] == watermark_r1

    # The next round must start its window at the restored watermark.
    r3 = await harvest_ok(harvester, store)
    run3 = store.get_run(r3)
    assert run3["window_from"] == watermark_r1
    assert store.published_state()["last_run_id"] == r3
    # Revoking round 3 restores round 1's pointers again (round 2 is
    # skipped: it was revoked before round 3 committed).
    assert store.revoke_run(r3)["revoked"] is True
    assert store.published_state()["watermark"] == watermark_r1
    assert store.published_state()["last_run_id"] == r1


async def test_failed_revoke_leaves_no_partial_state(
    harvester, store, repo, monkeypatch
):
    await harvest_ok(harvester, store)
    await asyncio.sleep(1.05)
    repo.touch_record("oai:demo:0001", title="误更正一")
    repo.touch_record("oai:demo:0002", title="误更正二")
    r2 = await harvest_ok(harvester, store)

    catalog_before = catalog_map(store)
    state_before = store.published_state()

    class ExplodingConnection:
        """Fails the second catalogue restore write inside the revoke."""

        def __init__(self, inner):
            self._inner = inner
            self._inserts = 0

        def execute(self, sql, *args, **kwargs):
            if sql.strip().startswith("INSERT INTO catalog"):
                self._inserts += 1
                if self._inserts == 2:
                    raise sqlite3.OperationalError("simulated disk error")
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    real_conn = store._conn
    monkeypatch.setattr(store, "_conn", ExplodingConnection(real_conn))
    with pytest.raises(sqlite3.OperationalError):
        store.revoke_run(r2)
    monkeypatch.setattr(store, "_conn", real_conn)

    # No half-restored catalogue, no watermark change, no revoke mark.
    assert catalog_map(store) == catalog_before
    state = store.published_state()
    assert state["watermark"] == state_before["watermark"]
    assert state["last_run_id"] == r2
    assert state["success_count"] == state_before["success_count"]
    assert store.get_run(r2)["revoked_at"] is None

    # The retained snapshot survives the failure, so a retry succeeds.
    assert store.revoke_run(r2)["revoked"] is True
    assert store.published_state()["watermark"] != state_before["watermark"]


async def test_revoke_and_snapshots_survive_restart(
    harvester, store, repo, tmp_path
):
    # NOTE: the `store` fixture owns its own db; use a separate file here so
    # we can close and reopen it to simulate a process restart.
    db = tmp_path / "restart.sqlite3"
    first = Store(db)
    run = first.create_run()
    first.set_window(run["id"], "2026-01-01T00:00:00Z")
    first.stage_records(
        run["id"],
        [
            {
                "identifier": "oai:kept:1",
                "title": "第一轮的记录",
                "creator": None,
                "datestamp": "2025-12-31T00:00:00Z",
                "deleted": False,
            }
        ],
        0,
    )
    assert first.commit_run(run["id"])["committed"]
    run2 = first.create_run()
    first.set_window(run2["id"], "2026-01-02T00:00:00Z")
    first.stage_records(
        run2["id"],
        [
            {
                "identifier": "oai:kept:1",
                "title": "第二轮误更正",
                "creator": None,
                "datestamp": "2026-01-01T12:00:00Z",
                "deleted": False,
            }
        ],
        0,
    )
    assert first.commit_run(run2["id"])["committed"]
    first.close()

    # After a restart the retained snapshot is still there: revoke works.
    second = Store(db)
    result = second.revoke_run(run2["id"])
    assert result["revoked"] is True
    item = catalog_map(second)["oai:kept:1"]
    assert item["title"] == "第一轮的记录"
    assert second.published_state()["watermark"] == "2026-01-01T00:00:00Z"
    second.close()

    # The revoked state itself is durable, and a new round can start from
    # the restored watermark after yet another restart.
    third = Store(db)
    assert third.get_run(run2["id"])["revoked_at"] is not None
    fresh = third.create_run()
    assert fresh["window_from"] == "2026-01-01T00:00:00Z"
    third.cancel_run(fresh["id"], "测试清理")
    third.close()


def test_legacy_database_is_migrated(tmp_path):
    """A database created before revoke support gets the new column."""
    db = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO runs(id, status, started_at) VALUES('r1', 'succeeded', 't')"
    )
    conn.commit()
    conn.close()

    store = Store(db)
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(runs)")}
    assert "revoked_at" in cols
    assert store.get_run("r1")["revoked_at"] is None
    # Revoke support tables exist as well.
    store._conn.execute("SELECT * FROM run_pre_state").fetchall()
    store._conn.execute("SELECT * FROM run_snapshots").fetchall()
    store.close()


# ------------------------------------------------------------------ web level


@pytest.fixture
def web_app(tmp_path: Path, demo_app, repo):
    def factory() -> OaiClient:
        return OaiClient(
            "http://demo.local/oai",
            transport=httpx.ASGITransport(app=demo_app),
        )

    return create_app(
        db_path=tmp_path / "web.sqlite3",
        oai_url="http://demo.local/oai",
        client_factory=factory,
    )


def poll_until(client: TestClient, predicate, attempts: int = 100) -> dict:
    import time

    for _ in range(attempts):
        payload = client.get("/api/status").json()
        if predicate(payload):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"condition not met; last={payload}")


def run_round(client: TestClient) -> dict:
    assert client.post("/api/harvest").status_code == 202
    return poll_until(
        client, lambda p: p["active"] is None and p["published"]["watermark"]
    )


def test_revoke_api_flow(web_app, repo):
    with TestClient(web_app) as client:
        first = run_round(client)
        r1 = first["published"]["last_run_id"]
        watermark_r1 = first["published"]["watermark"]

        repo.touch_record("oai:demo:0001", title="误更正的标题")
        repo.withdraw("oai:demo:0002")
        second = run_round(client)
        r2 = second["published"]["last_run_id"]
        assert r2 != r1
        assert "误更正的标题" in client.get("/").text

        # Only the latest successful round may be revoked.
        assert client.post(f"/api/runs/{r1}/revoke").status_code == 409
        assert client.post("/api/runs/no-such-id/revoke").status_code == 404

        response = client.post(f"/api/runs/{r2}/revoke")
        assert response.status_code == 200
        body = response.json()
        assert body["changed"] is True
        assert body["watermark"] == watermark_r1

        # Catalogue and watermark are restored; history marks the round.
        state = client.get("/api/status").json()
        assert state["published"]["watermark"] == watermark_r1
        assert state["published"]["last_run_id"] == r1
        assert state["revocable"]["id"] == r1
        page = client.get("/")
        assert "误更正的标题" not in page.text
        assert "数字人文中的文本挖掘方法" in page.text
        runs_page = client.get("/runs")
        assert "已撤销" in runs_page.text

        # Repeating the same id is a no-op, not a revoke of round 1.
        repeat = client.post(f"/api/runs/{r2}/revoke")
        assert repeat.status_code == 200
        assert repeat.json()["changed"] is False
        assert repeat.json()["status"] == "already_revoked"
        state = client.get("/api/status").json()
        assert state["published"]["last_run_id"] == r1


def test_revoke_api_refused_while_running(web_app, repo):
    with TestClient(web_app) as client:
        first = run_round(client)
        r1 = first["published"]["last_run_id"]

        repo.set_meta("slow_seconds", "5")
        assert client.post("/api/harvest").status_code == 202
        poll_until(client, lambda p: p["active"] is not None)

        response = client.post(f"/api/runs/{r1}/revoke")
        assert response.status_code == 409
        assert "进行" in response.json()["detail"]

        client.post("/api/harvest/cancel")
        poll_until(client, lambda p: p["active"] is None)
        repo.set_meta("slow_seconds", "0")

        assert client.post(f"/api/runs/{r1}/revoke").status_code == 200


def test_index_page_has_revoke_entry(web_app):
    with TestClient(web_app) as client:
        page = client.get("/")
        assert "撤销最近一次成功采集" in page.text
        # Nothing to revoke yet: the button starts disabled.
        assert 'id="btn-revoke"' in page.text

        run_round(client)
        page = client.get("/")
        assert "撤销最近一次成功采集" in page.text
        runs = client.get("/runs")
        assert "撤销本轮" in runs.text
