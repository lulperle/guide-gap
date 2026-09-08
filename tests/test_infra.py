"""Assertions on the synthesised template.

Not a deployment test -- this stack has never been deployed, and the README says
so. What it does check is the handful of properties that are expensive to discover
by deploying: that the bucket survives a stack deletion, that the cross-region
inference profile is actually permitted in every region it can route to, and that
the timeout leaves room for the alarm to fire before it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

aws_cdk = pytest.importorskip("aws_cdk", reason="pip install -e '.[infra]'")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra"))

from aws_cdk.assertions import Match, Template  # noqa: E402
from guide_gap_stack import DRAFT_MODEL, PROFILE_REGIONS, GuideGapStack  # noqa: E402


@pytest.fixture(scope="module")
def template() -> Template:
    app = aws_cdk.App()
    # A concrete account and region, not the ambient environment: the IAM ARNs are
    # built from them, so leaving them as tokens would make the policy assertions
    # pass against a template that never resolves.
    stack = GuideGapStack(
        app,
        "TestStack",
        env=aws_cdk.Environment(account="111122223333", region="us-west-2"),
    )
    return Template.from_stack(stack)


def test_the_artifact_bucket_survives_stack_deletion(template: Template):
    """Accepted drafts are documentation somebody edited. Every other resource
    here is reconstructible from the stack file; this one is not."""
    template.has_resource("AWS::S3::Bucket", {"DeletionPolicy": "Retain"})
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            }
        },
    )


def test_the_timeout_leaves_room_for_the_slow_alarm(template: Template):
    template.has_resource_properties("AWS::Lambda::Function", {"Timeout": 600})
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {"MetricName": "Duration", "Threshold": 480000},
    )


def test_the_inference_profile_is_permitted_in_every_region_it_routes_to(template: Template):
    """Allowing only the profile ARN produces an AccessDenied on some invocations
    and not others, which reads as a flaky endpoint rather than a policy error."""
    policies = template.find_resources("AWS::IAM::Policy")
    allowed = [
        resource
        for policy in policies.values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if statement.get("Action") == "bedrock:InvokeModel"
        for resource in _as_list(statement["Resource"])
        if isinstance(resource, str)
    ]
    model = DRAFT_MODEL.removeprefix("us.")
    for region in PROFILE_REGIONS:
        assert f"arn:aws:bedrock:{region}::foundation-model/{model}" in allowed


def test_bedrock_access_is_not_a_wildcard(template: Template):
    for policy in template.find_resources("AWS::IAM::Policy").values():
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            if statement.get("Action") == "bedrock:InvokeModel":
                assert "*" not in _as_list(statement["Resource"])


def test_the_job_is_scheduled_and_not_exposed(template: Template):
    template.has_resource_properties(
        "AWS::Events::Rule", {"ScheduleExpression": Match.string_like_regexp("cron")}
    )
    # A batch job with a public endpoint is an attack surface with no consumer.
    template.resource_count_is("AWS::ApiGateway::RestApi", 0)
    template.resource_count_is("AWS::Lambda::Url", 0)


def test_failures_and_slowdowns_both_alarm(template: Template):
    """A weekly job that failed silently produces no report, and an absent report
    looks the same as a week with no findings."""
    template.resource_count_is("AWS::CloudWatch::Alarm", 2)
    template.resource_count_is("AWS::SNS::Topic", 1)
    # No subscription in code: who gets paged is a team decision, and an address
    # here would end up in a public repository.
    template.resource_count_is("AWS::SNS::Subscription", 0)


def _as_list(value):
    return value if isinstance(value, list) else [value]
