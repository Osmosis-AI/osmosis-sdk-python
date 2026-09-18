# Managed development rollout servers

Run `osmosis dev server up` from a committed rollout folder containing `main.py`
and its dependencies. The platform installs the folder's dependencies and runs
committed HEAD. The command returns the server ID, rollout URL, API key and expiry;
`osmosis dev server list`, `logs ID` and `down ID` use the same managed lifecycle.

To request managed OpenSandbox credentials for one server:

```bash
osmosis dev server up --sandbox-environment opensandbox --ttl-hours 1
```

The platform must support per-server `sandbox_environment` selection before using
this option. The CLI requires the platform to acknowledge an explicit selection.
If an older platform ignores the field, the CLI requests teardown of the created
server and reports an error. Check `osmosis dev server list` for cleanup. If the
teardown request fails, the error includes `osmosis dev server down ID` for retry.
Omitting the option keeps the platform's existing provider default. `daytona` is
also supported. This selects managed sandbox credentials; the rollout server
continues to use the platform's existing ECS deployment.

The rollout's code must select a matching execution backend. For Harbor
OpenSandbox, select `EnvironmentType.OPENSANDBOX` and `use_server_proxy=True` so
commands and files use the native server relay instead of private sandbox pod IPs.
The platform launcher supplies `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` to the
host process. An explicitly configured customer key keeps its own connection
settings. Do not forward the native sandbox key into agent environment variables.
The native sandbox endpoint and key differ from the rollout URL and bearer key.

The command does not add missing provider capabilities or port a custom harness.
In particular, OpenCode registration and setup/model/verifier network phases for
the Qwen gVisor campaign require additional integration and an isolated canary.
Choosing OpenSandbox alone does not preserve those semantics.

The default lifetime is 24 hours. `--ttl-hours` can request a longer lifetime;
`--no-ttl` removes that CLI default but the platform still applies its server-side
ceiling (currently 336 hours). Stop unused servers explicitly.
