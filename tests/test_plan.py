from aac.plan import DEFAULT_PARAMS, ModelConfig, Plan, default_plan


def test_default_plan_and_hash():
    p = default_plan(["lightgbm", "logistic"], ["note", "id"])
    assert [m.family for m in p.models] == ["lightgbm", "logistic"]
    assert p.drop_columns == ["id", "note"]
    assert p.hash() == default_plan(["lightgbm", "logistic"], ["id", "note"]).hash()
    assert p.hash() != default_plan(["lightgbm"], ["id", "note"]).hash()
    assert len(p.hash()) == 16


def test_resolved_params_merge_defaults():
    p = Plan(name="x", models=[ModelConfig(family="lightgbm", params={"num_leaves": 7})])
    params = p.resolved_params(p.models[0])
    assert params["num_leaves"] == 7
    assert params["n_estimators"] == DEFAULT_PARAMS["lightgbm"]["n_estimators"]
    assert Plan.model_validate(p.model_dump()) == p
