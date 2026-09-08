"""CDK entry point.

    make layer && npx cdk synth      # template only, no credentials needed
    npx cdk deploy                   # not run for this repository -- see README

Environment comes from the CDK context rather than being hardcoded, so the same
template can go to a verification account and a production one without an edit.
"""

from __future__ import annotations

import aws_cdk as cdk
from guide_gap_stack import GuideGapStack

app = cdk.App()

GuideGapStack(
    app,
    "GuideGapStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-west-2",
    ),
    description="Weekly guide gap analysis: which page is missing, which exists but is unfindable",
)

cdk.Tags.of(app).add("project", "guide-gap")

app.synth()
