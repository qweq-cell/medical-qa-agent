#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Server 客户端自测工具 — 用官方 MCP 客户端连上你的 Server，看它到底暴露了什么。

用法:
    # 1) 只列工具（默认）
    python mcp-server/test_client.py --url http://localhost:8931/sse

    # 2) 列工具 + 真实调用一次质控（验证端到端）
    python mcp-server/test_client.py --url http://localhost:8931/sse --call

    # 3) stdio 模式（Claude Desktop / Cursor 用的那种）
    python mcp-server/test_client.py --stdio

前置:
    Server 已启动，例如:
        python mcp-server/server.py --transport sse --port 8931

输出:
    ① initialize 结果（Server 名 / 版本 / 协议版本）
    ② tools/list — 工具名、描述、入参 JSON Schema
    ③ （--call 时）真实调用一次并打印返回内容
"""
import argparse
import asyncio
import json
import os
import sys

SAMPLE_EMR = (
    "患者，男，56岁，因下腹痛就诊。既往：2型糖尿病5年，口服二甲双胍500mg每日三次。"
    "影像学：子宫肌瘤，大小约5cm。医嘱：己烯雌酚 2mg 每日一次。空腹血糖 9.8mmol/L。"
)


async def run(url: str | None, use_stdio: bool, do_call: bool):
    from mcp import ClientSession

    if use_stdio:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
        params = StdioServerParameters(command=sys.executable, args=[server_py])
        ctx = stdio_client(params)
        label = f"stdio: {sys.executable} {server_py}"
    else:
        from mcp.client.sse import sse_client

        ctx = sse_client(url)
        label = f"sse: {url}"

    print(f"连接方式: {label}\n")

    async with ctx as streams:
        # stdio_client / sse_client 返回的元组形状不同
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            si = init.serverInfo
            print("① initialize")
            print(f"   服务名:   {si.name}")
            print(f"   版本:     {si.version}")
            print(f"   协议版本: {init.protocolVersion}")
            if getattr(si, "instructions", None):
                print(f"   说明:     {si.instructions[:100]}")

            tools = await session.list_tools()
            print(f"\n② tools/list — 共 {len(tools.tools)} 个工具")
            for t in tools.tools:
                print(f"\n   ● {t.name}")
                desc = (t.description or "").strip()
                for line in desc.splitlines():
                    print(f"       {line}")
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                if props:
                    print("     参数:")
                    for pname, pinfo in props.items():
                        flag = "必填" if pname in required else "可选"
                        print(f"       - {pname} ({pinfo.get('type','?')}, {flag}): "
                              f"{pinfo.get('description','')}")

            if do_call:
                name = tools.tools[0].name
                arg = "emr_text" if "emr_text" in (tools.tools[0].inputSchema or {}).get("properties", {}) else None
                print(f"\n③ 真实调用 {name}")
                if arg is None:
                    print("   (无法确定入参名，跳过)")
                else:
                    result = await session.call_tool(name, {arg: SAMPLE_EMR})
                    print(f"   isError: {result.isError}")
                    for c in result.content:
                        text = getattr(c, "text", None)
                        if text:
                            print("   ---- 返回内容 ----")
                            for line in text.splitlines():
                                print(f"   {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8931/sse", help="SSE / streamable-http 地址")
    ap.add_argument("--stdio", action="store_true", help="用 stdio 方式连接本地 server.py")
    ap.add_argument("--call", action="store_true", help="列完工具后真实调用第一个工具")
    args = ap.parse_args()

    try:
        asyncio.run(run(args.url, args.stdio, args.call))
    except Exception as e:
        print(f"\n[失败] {type(e).__name__}: {e}")
        print("排查:")
        print("  1. Server 起了吗？  netstat -ano | findstr :8931")
        print("  2. 地址对吗？       /sse 对应 --transport sse")
        print("  3. 端口被占？       换 --port")
        sys.exit(1)


if __name__ == "__main__":
    main()
