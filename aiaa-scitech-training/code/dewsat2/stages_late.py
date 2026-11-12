"""Late stages called by run.py: score, checks, whatif, weekly, phase5, phase6."""

from __future__ import annotations


def score(status):
    import score_stage
    status.stage("score", state="running")
    score_stage.run(status)
    status.stage("score", state="done")


def checks(status):
    import checks_stage
    status.stage("checks", state="running")
    checks_stage.run(status)
    status.stage("checks", state="done")


def whatif(status):
    import whatif_stage
    status.stage("whatif", state="running")
    whatif_stage.run(status)
    status.stage("whatif", state="done")


def weekly(status):
    import weekly_stage
    status.stage("weekly", state="running")
    weekly_stage.run(status)
    status.stage("weekly", state="done")


def phase5(status):
    import phase5_stage
    status.stage("phase5", state="running")
    phase5_stage.run(status)
    status.stage("phase5", state="done")


def phase6(status):
    import phase6_stage
    status.stage("phase6", state="running")
    phase6_stage.run(status)
    status.stage("phase6", state="done")
