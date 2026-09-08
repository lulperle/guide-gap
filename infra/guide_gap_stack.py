"""The stack that runs the analysis on a schedule.

This is a batch job, not a service. Nobody calls it; it wakes up weekly, reads the
ticket export and the guide, and leaves a report and some draft pages in S3 for a
human to accept or reject. Modelling it as an API would add a public surface, an
authorizer and a latency budget to something whose entire consumer is a person
reading a markdown file on Monday morning.

Three choices here are worth arguing with rather than skimming.

**Lambda, with the ceiling written down.** The pass is sequential Bedrock calls --
about 66 for 22 tickets, three minutes -- so Lambda fits today with room to spare.
It does not fit at a few thousand tickets, because the calls grow with the batch
and the hard limit is fifteen minutes. `MAX_TICKETS_PER_RUN` makes that boundary an
explicit input rather than a surprise timeout, and the alarm on duration fires long
before the wall.

**The bucket is retained.** Deleting this stack must not delete the accepted
drafts; they are documentation somebody edited. Every other resource here is
reconstructible from this file, so it is the one thing that gets a retain policy.

**Least privilege on the model, spelled out per region.** See the comment on the
IAM policy -- a cross-region inference profile does not work with only its own ARN
allowed, and that failure looks like an intermittent AccessDenied on some
invocations and not others.
"""

from __future__ import annotations

from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as actions
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
from constructs import Construct

# The `us.` prefix is a cross-region inference profile rather than a plain model id.
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DRAFT_MODEL = "us.anthropic.claude-sonnet-5"

# Where a cross-region profile is allowed to route. The profile decides at request
# time, so every one of these has to be permitted or the run fails on whichever
# invocation happened to land in the region that was left out.
PROFILE_REGIONS = ("us-east-1", "us-east-2", "us-west-2")

# Runtime cap, not a scaling strategy. Fifteen minutes is Lambda's ceiling and this
# job is sequential, so the honest move is to bound the batch and say what happens
# when it is exceeded: the run stops and reports how many were left.
MAX_TICKETS_PER_RUN = 500


class GuideGapStack(Stack):
    """Weekly gap analysis: EventBridge -> Lambda -> S3, with an alarm."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        schedule: events.Schedule | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.artifacts = s3.Bucket(
            self,
            "Artifacts",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            # Drafts are documentation somebody edited. Losing them because a stack
            # was torn down would be the worst failure this stack could have, and it
            # would be silent.
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    # Old runs are kept for comparison -- the whole point is
                    # measuring whether the guide got better -- but not forever.
                    id="expire-old-runs",
                    prefix="runs/",
                    expiration=Duration.days(365),
                    noncurrent_version_expiration=Duration.days(90),
                )
            ],
        )

        # numpy is not in the Lambda runtime, and a layer keeps the function asset
        # small enough to read in the console. Built by `make layer`, not vendored
        # into git -- see infra/README.md.
        dependencies = lambda_.LayerVersion(
            self,
            "Dependencies",
            code=lambda_.Code.from_asset("infra/layer"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="numpy and pyyaml for the analysis function",
        )

        self.analysis = lambda_.Function(
            self,
            "Analysis",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.run",
            code=lambda_.Code.from_asset("infra/function"),
            layers=[dependencies],
            # Ten minutes, deliberately under the fifteen-minute maximum so the
            # duration alarm has somewhere to fire before the hard timeout, which
            # produces no report at all.
            timeout=Duration.minutes(10),
            # The corpus is embedded in memory as float32 vectors. A few thousand
            # sections is single-digit megabytes; the memory here is for CPU share,
            # since Lambda scales both together and the numpy matmul benefits.
            memory_size=1024,
            environment={
                "ARTIFACT_BUCKET": self.artifacts.bucket_name,
                "GUIDE_GAP_EMBED_MODEL": EMBED_MODEL,
                "GUIDE_GAP_DRAFT_MODEL": DRAFT_MODEL,
                "MAX_TICKETS_PER_RUN": str(MAX_TICKETS_PER_RUN),
            },
            log_retention=logs.RetentionDays.THREE_MONTHS,
        )

        self.artifacts.grant_read_write(self.analysis, "runs/*")
        # The embedding cache. Separated from the artifacts prefix because it is
        # regenerable and the artifacts are not, so they get different lifecycles
        # and a reader can tell at a glance which is safe to delete.
        self.artifacts.grant_read_write(self.analysis, "cache/*")

        self.analysis.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                # Two ARN shapes, and both are required.
                #
                # A cross-region inference profile is invoked by its own ARN, but
                # Bedrock then calls the underlying foundation model in whichever
                # region it routed to, and that call is authorised against the
                # caller's identity too. Allowing only the profile ARN produces an
                # AccessDenied on some invocations and not others, which reads as a
                # throttle or a flaky endpoint rather than as a policy error.
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/{EMBED_MODEL}",
                    f"arn:aws:bedrock:{self.region}:{self.account}"
                    f":inference-profile/{DRAFT_MODEL}",
                    *[
                        f"arn:aws:bedrock:{region}::foundation-model/"
                        f"{DRAFT_MODEL.removeprefix('us.')}"
                        for region in PROFILE_REGIONS
                    ],
                ],
            )
        )

        events.Rule(
            self,
            "Weekly",
            # Weekly, because the thing being measured moves on the timescale of
            # somebody writing a guide page. Running it hourly would produce a
            # graph that looks busy and a backlog nobody reads.
            schedule=schedule or events.Schedule.cron(week_day="MON", hour="2", minute="0"),
            targets=[targets.LambdaFunction(self.analysis, retry_attempts=1)],
        )

        self.alarms = sns.Topic(self, "Alarms", display_name="guide-gap alarms")
        # No subscription in code on purpose: who gets paged is a team decision and
        # putting an address here means it ends up in a public repository.

        cloudwatch.Alarm(
            self,
            "AnalysisFailed",
            metric=self.analysis.metric_errors(period=Duration.hours(1)),
            threshold=1,
            evaluation_periods=1,
            # A weekly job that failed silently means the report is simply absent,
            # and an absent report looks the same as a week with no findings.
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="The weekly guide-gap analysis did not complete.",
        ).add_alarm_action(actions.SnsAction(self.alarms))

        cloudwatch.Alarm(
            self,
            "AnalysisSlow",
            metric=self.analysis.metric_duration(period=Duration.hours(1)),
            # Eight minutes against a ten-minute timeout: the warning that the batch
            # has grown past what a sequential Lambda pass can finish, while there
            # is still time to move it to Fargate before it starts timing out.
            threshold=Duration.minutes(8).to_milliseconds(),
            evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="The analysis is approaching its timeout; the batch has grown.",
        ).add_alarm_action(actions.SnsAction(self.alarms))
