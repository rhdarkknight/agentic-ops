"""CompSkillBench-style eval set adapted for Hermes' skill library.

30 compositional queries spanning multiple categories:
  - 10 easy  (2 sub-tasks, 2 distinct skill domains)
  - 10 medium (3 sub-tasks, 3 distinct skill domains)
  - 10 hard  (4+ sub-tasks, 4+ distinct skill domains)

Each query has a `ground_truth_skills` list — expected skill *names* in
the order they should appear in the plan. Categories listed in
`ground_truth_categories` for CatR@k style eval (any skill from that
category counts).

Following SkillWeaver's metrics:
  - DA   (decomposition accuracy): exact sub-task count match
  - DA±1 : predicted sub-tasks within ±1 of ground truth
  - SR@1 : ground-truth #1 skill in top-1 retrieval for at least one sub-task
  - SR@3 : all ground-truth skills retrieved within top-3 across all sub-tasks
  - CatR@3: any skill from each ground-truth category in top-3 retrieval
  - PlanEM: plan exactly matches ground-truth ordering
"""

from __future__ import annotations
from typing import Any


def _q(query, expected_skills, expected_categories, n_subtasks):
    return {
        "query": query,
        "ground_truth_skills": expected_skills,
        "ground_truth_categories": expected_categories,
        "n_subtasks": n_subtasks,
    }


EASY = [
    _q(
        "Deploy a smart contract to Base and verify it on the explorer",
        ["base", "ethers-js"],
        ["base", "ethers-js"],
        2,
    ),
    _q(
        "Compose a tweet and post it to X",
        ["xitter", "xurl"],
        ["xitter", "xurl"],
        2,
    ),
    _q(
        "Swap SOL for USDC on Orca and stake the USDC on Kamino",
        ["orca", "kamino"],
        ["orca", "kamino"],
        2,
    ),
    _q(
        "Audit a Solidity contract and run slither on it",
        ["solidity-auditor", "slither"],
        ["solidity-auditor", "slither"],
        2,
    ),
    _q(
        "Read my Gmail inbox and forward specific messages",
        ["gmail-app-password-setup", "himalaya"],
        ["email", "email"],
        2,
    ),
    _q(
        "List my crypto positions on Hyperliquid and Jupiter",
        ["hyperliquid", "jupiter_health_monitor"],
        ["hyperliquid", "jupiter"],
        2,
    ),
    _q(
        "Open a Safe multisig and submit a transaction",
        ["safe", "safe"],
        ["safe", "safe"],
        2,
    ),
    _q(
        "Configure a Breeze RMM alert and acknowledge it",
        ["mcp-breeze-manage-notification-channels", "mcp-breeze-acknowledge-network-device"],
        ["msp", "msp"],
        2,
    ),
    _q(
        "Generate a Solidity contract and compile with Foundry",
        ["foundry", "openzeppelin"],
        ["foundry", "openzeppelin"],
        2,
    ),
    _q(
        "Search the web and extract content from a page",
        ["web-search", "web-extract"],
        ["web", "web"],
        2,
    ),
]


MEDIUM = [
    _q(
        "1. Find the cheapest gas on L2, 2. bridge USDC via deBridge, 3. supply to Aave",
        ["debridge", "aave"],
        ["debridge", "aave"],
        3,
    ),
    _q(
        "Scan the fleet for high CPU, alert on it, then auto-remediate by restarting the service",
        ["mcp-breeze-query-devices", "mcp-breeze-apply-cis-remediation", "mcp-breeze-manage-services"],
        ["msp", "msp", "msp"],
        3,
    ),
    _q(
        "Create a Solana program, deploy it, and verify the program on chain",
        ["solana-kit", "surfpool"],
        ["solana-kit", "solana-kit"],
        3,
    ),
    _q(
        "Build an MCP server for an API and register it with the gateway",
        ["mcp-integration", "hermes-agent"],
        ["mcp", "hermes-agent"],
        3,
    ),
    _q(
        "Take a screenshot of a webpage, OCR it, then summarize the text",
        ["browser-vision", "transcription-summary"],
        ["browser", "media"],
        3,
    ),
    _q(
        "Compose a multi-asset LP on Meteora, then monitor it",
        ["meteora", "kamino"],
        ["meteora", "kamino"],
        3,
    ),
    _q(
        "Scrape a Next.js site, extract structured data, write to Google Sheets",
        ["nextjs-data-extraction", "google-workspace"],
        ["web-scraping", "productivity"],
        3,
    ),
    _q(
        "Set up a Breeze patch policy, approve critical patches, then verify install",
        ["mcp-breeze-manage-patches", "mcp-breeze-apply-cis-remediation"],
        ["msp", "msp"],
        3,
    ),
    _q(
        "Create a cron job, attach a skill, then verify it runs",
        ["cron-job-creation", "cron-job-execution-mechanism"],
        ["cron", "cron"],
        3,
    ),
    _q(
        "Mint an NFT via Metaplex and list it on Tensor",
        ["metaplex", "sanctum"],
        ["metaplex", "sanctum"],
        3,
    ),
]


HARD = [
    _q(
        "Set up a Hyper-V VM, install pihole, configure it as DNS sinkhole, then verify it blocks ads",
        ["hyperv-vm-provision-pihole", "lan-dns-sinkhole-deployment", "beelinkhost-hyperv", "qnap-nas-operations"],
        ["hyperv", "lan", "hyperv", "msp"],
        4,
    ),
    _q(
        "Build a dapp on Optimism, deploy it, verify on explorer, then monitor it",
        ["optimism", "openzeppelin", "wagmi", "tenderly"],
        ["optimism", "openzeppelin", "wagmi", "tenderly"],
        4,
    ),
    _q(
        "Pull a paper from arxiv, summarize it, generate a presentation, then email it to me",
        ["paper-watch", "transcription-summary", "presenton-marketing-decks", "himalaya"],
        ["research", "media", "productivity", "email"],
        4,
    ),
    _q(
        "Compose a multisig transaction, simulate it on Tenderly, then execute via Safe",
        ["safe", "tenderly", "safe", "wallet"],
        ["safe", "tenderly", "safe", "wallet"],
        4,
    ),
    _q(
        "Discover a Solana program, audit it with Sec3, then deploy a fix",
        ["solana-kit", "certora", "pinocchio"],
        ["solana-kit", "certora", "pinocchio"],
        4,
    ),
    _q(
        "Pull a Trello board, summarize activity, draft a status update, and post to Slack",
        ["composio-sdk-install", "transcription-summary", "hermes-composio", "jira-tools"],
        ["composio", "media", "composio", "productivity"],
        4,
    ),
    _q(
        "Crawl a site, extract emails, run PII removal on each, then write report",
        ["web-scraping", "pii-removal-automation", "create-ms-office-files"],
        ["web-scraping", "privacy", "productivity"],
        4,
    ),
    _q(
        "Install Breeze RMM, enroll 3 devices, configure monitoring, then verify alerts via Telegram",
        ["breeze-rmm-integration", "mcp-breeze-send-deployment-invites", "mcp-breeze-list-configuration-policies", "mcp-breeze-manage-alert-rules"],
        ["msp", "msp", "msp", "notification"],
        4,
    ),
    _q(
        "Build a hyperliquid strategy, backtest it, then run paper trade",
        ["hyperliquid", "academic-benchmarks", "drift"],
        ["hyperliquid", "academic-benchmarks", "drift"],
        4,
    ),
    _q(
        "Wire up Composio Gmail, fetch unread, summarize with hindsight, then draft reply",
        ["hermes-composio", "hindsight-recall", "himalaya"],
        ["composio", "memory", "email"],
        4,
    ),
]


EVAL_SET: list[dict[str, Any]] = EASY + MEDIUM + HARD


def by_difficulty() -> dict[str, list[dict[str, Any]]]:
    return {"easy": EASY, "medium": MEDIUM, "hard": HARD}