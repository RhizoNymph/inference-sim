"""Minimal disaggregated prefill/decode proxy for vLLM's NixlConnector.

Follows vLLM's NIXL toy proxy: each /v1/completions request goes to the
prefill server as a non-streaming, max_tokens=1 request asking for remote
decode; the returned kv_transfer_params are forwarded to the decode server,
which pulls the KV cache over NIXL and streams the completion back.

Run inside the vLLM venv (fastapi, httpx and uvicorn ship with vLLM):
    python disagg_proxy.py --prefill http://10.1.1.69:8100 --decode http://10.1.1.68:8200 --port 8000
"""

import argparse
import copy
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("disagg_proxy")


def build_app(prefill_url: str, decode_url: str) -> FastAPI:
    app = FastAPI()
    client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    @app.get("/health")
    async def health() -> Response:
        for url in (prefill_url, decode_url):
            response = await client.get(f"{url}/health")
            if response.status_code != 200:
                return Response(status_code=503)
        return Response(status_code=200)

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        response = await client.get(f"{decode_url}/v1/models")
        return JSONResponse(response.json(), status_code=response.status_code)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        body = await request.json()

        prefill_body = copy.deepcopy(body)
        prefill_body["max_tokens"] = 1
        prefill_body["stream"] = False
        prefill_body.pop("stream_options", None)
        prefill_body.pop("min_tokens", None)
        prefill_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        prefill_response = await client.post(f"{prefill_url}/v1/completions", json=prefill_body)
        if prefill_response.status_code != 200:
            logger.error(
                "prefill failed status=%s body=%s", prefill_response.status_code, prefill_response.text[:500]
            )
            return Response(prefill_response.content, status_code=prefill_response.status_code)
        kv_transfer_params = prefill_response.json().get("kv_transfer_params")

        decode_body = copy.deepcopy(body)
        if kv_transfer_params:
            decode_body["kv_transfer_params"] = kv_transfer_params

        if not body.get("stream"):
            decode_response = await client.post(f"{decode_url}/v1/completions", json=decode_body)
            return Response(
                decode_response.content,
                status_code=decode_response.status_code,
                media_type="application/json",
            )

        async def stream():
            async with client.stream("POST", f"{decode_url}/v1/completions", json=decode_body) as response:
                async for chunk in response.aiter_raw():
                    yield chunk

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(build_app(args.prefill, args.decode), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
