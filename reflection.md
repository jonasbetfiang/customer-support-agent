Reflection:

I jonas, Please find my reflection notice on the assignment That I found particularly stressfull during the deployment step. but the rest was pretty straitforward.

Design decision: I chose to give calculate_loyalty_discount a graceful fallback
path rather than letting it fail outright if the Code Interpreter sandbox is
unavailable. When the primary execution path throws, the tool still returns a
usable (if less precise) discount based on tier alone, rather than returning
an error the agent would have to explain to the customer. In a customer-facing
support agent, a degraded-but-useful answer is almost always better than a
hard failure, especially for a calculation customers care about in the
moment.

Challenge encountered: The hardest part of this project wasn't the agent code
itself but the deployment path. AWS CodeBuild's container build repeatedly
stopped in the PROVISIONING phase with no logs, which I traced to the lab
account's restriction on privileged container builds. I switched to
--local-build to build the Docker image locally instead, which required
enabling CPU virtualization in my laptop's BIOS, then troubleshooting a
sequence of local Docker networking and ECR authentication issues before the
push succeeded. Once deployed, I hit a second class of problem: IAM
permissions. The auto-generated execution role covered the runtime's core
needs but not the Memory or Knowledge Base resources I'd created separately
in the console, so I added scoped inline policies for bedrock-agentcore:GetMemory
(and related memory actions) and bedrock:Retrieve once I found the exact
denied action in the CloudWatch logs. The one issue I could not fully resolve
was the browser tool, which triggers a repeating "cannot create weak
reference to NoneType" error from anyio immediately after being invoked —
confirmed via logs to be a real runtime crash, not a permissions issue, and
consistent with a Python 3.14 / anyio compatibility gap in the deployed
container's async task handling rather than a bug in my own tool wiring.

Production extension: For production, I'd add per-tool retry/circuit-breaker
logic (especially around the browser and Code Interpreter tools, which are
the least reliable), pin dependency versions explicitly rather than relying
on latest-compatible resolution, and expand the CloudWatch alarm into a small
dashboard tracking error rate by tool name, not just a blanket error count,
so a single flaky tool doesn't get lost in the aggregate.
