from app.pipeline.query_understander import understand


def test_where_is_robotics_lab() -> None:
    assert understand("where is the robotics lab").best_entity == "robotics lab"


def test_tell_me_about_computing_lab() -> None:
    assert understand("tell me about the computing lab").best_entity == "computing lab"


def test_need_to_find_cs_department() -> None:
    assert understand("I need to find the CS department").best_entity == "CS department"


def test_take_me_to_room_navigation() -> None:
    understood = understand("take me to room 214")
    assert understood.best_entity == "room 214"
    assert understood.query_type == "navigation"


def test_person_prefix_detected() -> None:
    understood = understand("where does Dr Ahmed have office hours")
    assert "Ahmed" in understood.best_entity
    assert understood.has_person_prefix is True


def test_show_me_library() -> None:
    assert understand("can you show me the library").best_entity == "library"


def test_looking_for_engineering_faculty() -> None:
    assert understand("I'm looking for the engineering faculty").best_entity == "engineering faculty"


def test_router_entity_trusted_when_confident() -> None:
    understood = understand("where is the lab", router_entity="Robotics Lab", router_confidence=0.75)
    assert understood.best_entity == "Robotics Lab"


def test_router_entity_ignored_when_low_confidence() -> None:
    understood = understand("where is the robotics lab", router_entity="Wrong Lab", router_confidence=0.74)
    assert understood.best_entity == "robotics lab"
