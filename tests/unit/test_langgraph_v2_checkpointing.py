from app.langgraph_v2.checkpointing import thread_checkpoint_config, thread_id_for


def test_thread_checkpoint_config_only_sets_thread_id() -> None:
    assert thread_checkpoint_config(thread_id="thread-1") == {
        "configurable": {"thread_id": "thread-1"}
    }


def test_thread_id_is_collision_free_across_trusted_scope_parts() -> None:
    conversation_id = "00000000-0000-0000-0000-000000000001"
    identities = {
        thread_id_for("tenant:a", "subject", "linear", conversation_id),
        thread_id_for("tenant", "a:subject", "linear", conversation_id),
        thread_id_for("tenant:a", "other-subject", "linear", conversation_id),
        thread_id_for("tenant:a", "subject", "agent", conversation_id),
        thread_id_for(
            "tenant:a",
            "subject",
            "linear",
            "00000000-0000-0000-0000-000000000002",
        ),
    }

    assert len(identities) == 5
