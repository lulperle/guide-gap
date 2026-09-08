"""The only file that talks to AWS.

Deliberately thin and deliberately alone. Everything above this layer -- coverage
classification, clustering, metrics -- is pure functions over vectors and text, so
it is testable without credentials and without a network. That split is the reason
the test suite runs in CI at all.

Two models, chosen for different jobs:

- Titan Text Embeddings V2 for retrieval. Multilingual, so a Japanese ticket and
  a Japanese guide page land near each other, which is the whole premise here.
- Claude via the Converse API for drafting. Converse rather than InvokeModel
  because the request shape is the same across providers, so swapping the model is
  a config change rather than a rewrite of the call site.

No agent framework. This does one pass over a batch of tickets on a schedule; it
does not need a tool-calling loop, and an agent that cannot show why it grouped
two tickets together is worse here than arithmetic that can.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-west-2")

EMBED_MODEL = os.environ.get("GUIDE_GAP_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
# The `us.` prefix is a cross-region inference profile, not a plain model id.
# Without it this id is not invokable in most regions.
DRAFT_MODEL = os.environ.get("GUIDE_GAP_DRAFT_MODEL", "us.anthropic.claude-sonnet-5")

EMBED_DIMENSIONS = 1024

# Retries are left to botocore, which already implements them properly. Rolling
# our own on top would stack two backoffs and turn one throttle into a minute of
# waiting.
_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})


@dataclass
class Usage:
    """What the run cost. Reported, not estimated."""

    embed_calls: int = 0
    embed_tokens: int = 0
    draft_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    # Calls that hit the token cap. Tracked because a truncated response is not an
    # error -- the API returns 200 and a well-formed message -- and on a model that
    # emits reasoning before text, a truncated response can contain no text at all.
    # A caller that ignores this reads the empty string as an answer.
    truncated: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: Usage) -> None:
        self.embed_calls += other.embed_calls
        self.embed_tokens += other.embed_tokens
        self.draft_calls += other.draft_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.seconds += other.seconds
        self.truncated += other.truncated
        self.errors.extend(other.errors)


class Bedrock:
    """Embeddings and text generation, with usage accounted for."""

    def __init__(self, region: str = REGION) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=region, config=_CONFIG)
        self.usage = Usage()

    def embed(self, text: str) -> list[float]:
        """Embed one string. Titan takes a single input per call, not a batch."""
        started = time.monotonic()
        response = self._client.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps({"inputText": text, "dimensions": EMBED_DIMENSIONS}),
        )
        payload = json.loads(response["body"].read())
        self.usage.embed_calls += 1
        self.usage.embed_tokens += payload.get("inputTextTokenCount", 0)
        self.usage.seconds += time.monotonic() - started
        return payload["embedding"]

    def embed_all(self, texts: list[str]) -> list[list[float]]:
        """Embed a list, one call each.

        Sequential on purpose. The batch here is a few dozen documents on a
        schedule, so the latency is irrelevant, and concurrency would trade a
        clear failure for a partially-populated index that looks fine.
        """
        return [self.embed(t) for t in texts]

    def draft(self, system: str, prompt: str, max_tokens: int = 2000) -> str:
        """One generation call. No conversation state, no tools."""
        started = time.monotonic()
        try:
            response = self._client.converse(
                modelId=DRAFT_MODEL,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                # No `temperature`. Claude Sonnet 5 rejects it outright --
                # "`temperature` is deprecated for this model" as a
                # ValidationException, not a warning -- so the usual
                # temperature=0 lever for reproducibility is simply not available
                # here. That is why the eval measures stability by repeating the
                # run instead of assuming the output is pinned.
                inferenceConfig={"maxTokens": max_tokens},
            )
        except ClientError as exc:
            # A failed draft is one missing page, not a failed run. The batch
            # keeps going and the error is reported in the usage record, because
            # a pipeline that dies on ticket 14 of 20 has told you nothing about
            # the other six.
            self.usage.errors.append(f"draft failed: {exc.response['Error']['Code']}")
            self.usage.seconds += time.monotonic() - started
            return ""

        usage = response.get("usage", {})
        self.usage.draft_calls += 1
        self.usage.input_tokens += usage.get("inputTokens", 0)
        self.usage.output_tokens += usage.get("outputTokens", 0)
        self.usage.seconds += time.monotonic() - started
        if response.get("stopReason") == "max_tokens":
            self.usage.truncated += 1

        # Only `text` blocks. A `reasoningContent` block carries no answer, and on
        # a truncated response it can be the *only* block present -- which is how a
        # too-small maxTokens turns into an empty string that reads like a reply.
        return "".join(
            block.get("text", "") for block in response["output"]["message"]["content"]
        ).strip()
