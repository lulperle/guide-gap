# infra

CDK stack for running the analysis weekly. **This has never been deployed.** It
synthesises, and `tests/test_infra.py` asserts on the resulting template, but no
resources exist in any account and no numbers anywhere in this repository came from
running it in AWS. The measured results in the top-level README came from running
`scripts/analyse.py` and `evals/run.py` locally against Bedrock.

```
make layer            # build the Lambda layer for the Lambda runtime, not this laptop
make synth            # template only, no credentials needed
.venv/bin/python -m pytest tests/test_infra.py
```

## Shape

```
EventBridge (weekly, Mon 02:00 UTC)
  -> Lambda  guide_gap.pipeline.analyse
       reads   s3://…/guide/*.md, s3://…/input/tickets.yaml, s3://…/cache/
       writes  s3://…/runs/<stamp>/report.txt, drafts/*.md, result.json
  -> CloudWatch alarms -> SNS topic (no subscription in code)
```

## Decisions worth arguing with

**Drafts stop in S3.** Nothing here writes to the guide. A generated page reaching
users without a person reading it first is the failure the drafting prompt is built
to avoid, and an automated commit would put that one IAM grant away.

**Lambda, with the ceiling written down.** The pass is sequential Bedrock calls --
about 66 for 22 tickets, three minutes measured. It fits today; it does not fit at
a few thousand tickets, because Lambda stops at fifteen minutes.
`MAX_TICKETS_PER_RUN` makes that an explicit input rather than a surprise timeout,
and the duration alarm fires at eight minutes against a ten-minute timeout, so
there is warning before there is an outage. The migration when it comes is Fargate.

**The bucket is retained on stack deletion.** Accepted drafts are documentation
somebody edited; every other resource here is reconstructible from
`guide_gap_stack.py`.

**Bedrock permissions are per region, on purpose.** `us.anthropic.claude-sonnet-5`
is a cross-region inference profile. It is invoked by the profile ARN, but Bedrock
then calls the underlying foundation model in whichever region it routed to, and
that call is authorised against the same identity. Allowing only the profile ARN
produces `AccessDenied` on some invocations and not others -- which reads as a
throttle or a flaky endpoint, not as a policy error. So the policy lists the
foundation-model ARN in every region the profile can route to, and
`test_infra.py::test_the_inference_profile_is_permitted_in_every_region_it_routes_to`
fails if one is dropped.

**No SNS subscription in code.** Who gets paged is a team decision, and an email
address here would be committed to a public repository.

## What is missing

- Never deployed, so nothing here is validated against a real IAM evaluation, a
  real cold start, or a real S3 event. Read it as a design, not as evidence.
- No VPC. The function talks only to Bedrock and S3 over public endpoints. In a
  government-cloud setting that is the wrong answer and it would need a VPC with
  interface endpoints for both, plus the subnet and security-group wiring.
- No KMS CMK. `S3_MANAGED` encryption is the default here; a customer-managed key
  with a rotation policy is what an audited environment would ask for.
- The ticket export lands in `input/tickets.yaml` by some means outside this stack.
  Wiring that up is where a real deployment would start, and the shape of it
  depends entirely on which ticketing system is upstream.
