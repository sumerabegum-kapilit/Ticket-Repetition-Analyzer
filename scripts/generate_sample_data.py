"""Build a synthetic tickets dataset shaped like the real MongoDB schema, with
genuine near-duplicate wording per recurring issue (not copy-pasted text) so
the embedding/clustering pipeline has something real to prove itself against.

Run: python scripts/generate_sample_data.py
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

random.seed(42)

NOW = datetime.now(timezone.utc)
WINDOW_DAYS = 90

DEPARTMENTS = ["Finance", "Operations", "HR", "Sales", "Engineering", "Customer Support", "Legal"]
BRANCHES = ["Head Office", "North Branch", "South Branch", "West Wing", "3rd Floor Annex"]
REPORTER_NAMES = [
    "A. Sharma", "R. Iyer", "M. Fernandes", "K. Rao", "S. Gupta", "P. Nair",
    "T. Das", "V. Menon", "N. Kulkarni", "J. Pillai", "L. D'Souza", "H. Verma",
]

# closed/reopened tickets get a closed_at timestamp so resolution-time
# insights (avg/fastest/slowest) have something real to compute over -
# higher-priority tickets resolve faster on average, same as a real queue.
STATUS_WEIGHTS = {"open": 0.22, "in_progress": 0.13, "closed": 0.58, "reopened": 0.07}
PRIORITY_RESOLUTION_HOURS = {"critical": (1, 12), "high": (4, 30), "medium": (12, 96), "low": (24, 200)}


def _pick_status() -> str:
    labels = list(STATUS_WEIGHTS.keys())
    probs = list(STATUS_WEIGHTS.values())
    return random.choices(labels, weights=probs, k=1)[0]


def _maybe_close(ticket: dict, created: datetime, status: str) -> None:
    """A closed or reopened ticket was resolved at some point - add that
    timestamp, capped at "now" so a recently-created ticket can't show a
    resolution time in the future."""
    if status not in ("closed", "reopened"):
        return
    lo, hi = PRIORITY_RESOLUTION_HOURS[ticket["priority"]]
    closed = created + timedelta(hours=random.uniform(lo, hi))
    if closed > NOW:
        closed = NOW
    ticket["closed_at"] = closed.isoformat().replace("+00:00", "Z")

# Each recurring issue: category/domain + several DIFFERENTLY WORDED subject/description
# variants (this is what the embedding model has to recognize as "the same issue"),
# a priority distribution, a rough target count, and a volume shape over the 90-day window.
ISSUES = [
    {
        "category": "Access", "domain": "IT Support",
        "subjects": [
            "Unable to reset my password",
            "Password reset not working",
            "Locked out of my account after password change",
            "Cannot log in - account locked",
            "Need urgent password reset, account locked",
            "Forgot password and reset link not arriving",
        ],
        "descriptions": [
            "I tried resetting my password from the login page but the reset link never arrived in my inbox.",
            "My account got locked after 3 failed login attempts and I cannot self-service reset it.",
            "The 'forgot password' flow just spins and never sends an email to my registered address.",
            "I changed my password yesterday and now the new one is rejected as invalid on every attempt.",
        ],
        "priority_weights": {"low": 0.1, "medium": 0.55, "high": 0.3, "critical": 0.05},
        "count": 63, "shape": "flat",
    },
    {
        "category": "Network", "domain": "IT Support",
        "subjects": [
            "VPN connection keeps dropping",
            "VPN disconnects every few minutes",
            "Unable to stay connected to VPN",
            "VPN drops when working from home",
            "Constant VPN disconnections since this morning",
            "VPN client crashes and disconnects randomly",
        ],
        "descriptions": [
            "The VPN client disconnects every 5-10 minutes and I have to manually reconnect each time.",
            "Since this morning my VPN session drops randomly, interrupting file transfers and calls.",
            "Working from home and the VPN keeps timing out, even on a stable home network connection.",
            "VPN was fine last week; now it drops constantly and reconnecting takes several attempts.",
        ],
        "priority_weights": {"low": 0.05, "medium": 0.35, "high": 0.45, "critical": 0.15},
        "count": 47, "shape": "rising",
    },
    {
        "category": "Network", "domain": "IT Support",
        "subjects": [
            "Wifi disconnects intermittently on 3rd floor",
            "Weak wifi signal, keeps dropping",
            "Wireless network unstable near meeting rooms",
            "Wifi keeps cutting out throughout the day",
            "Intermittent wifi outage - 3rd floor annex",
        ],
        "descriptions": [
            "The wifi on the 3rd floor drops every hour or so, especially near the meeting rooms.",
            "Wireless signal is weak and the connection keeps cutting out during video calls.",
            "Multiple people on our floor are reporting the wifi disconnecting a few times a day.",
        ],
        "priority_weights": {"low": 0.15, "medium": 0.6, "high": 0.2, "critical": 0.05},
        "count": 41, "shape": "flat",
    },
    {
        "category": "Hardware", "domain": "IT Support",
        "subjects": [
            "Printer not responding on 3rd floor",
            "Printer stuck, won't print anything",
            "Unable to print - printer offline",
            "3rd floor printer showing offline status",
            "Print jobs stuck in queue, nothing prints",
        ],
        "descriptions": [
            "The shared printer on the 3rd floor shows as offline even though it's powered on.",
            "My print jobs sit in the queue forever and never actually print.",
            "Tried printing from two different laptops, printer isn't responding to either.",
        ],
        "priority_weights": {"low": 0.3, "medium": 0.55, "high": 0.15, "critical": 0.0},
        "count": 34, "shape": "falling",
    },
    {
        "category": "Software", "domain": "Business Apps",
        "subjects": [
            "ERP module running slow during month-close",
            "ERP system extremely slow at month end",
            "Finance module lagging badly this week",
            "ERP taking forever to load reports during closing",
            "Severe slowness in ERP during month-close activities",
        ],
        "descriptions": [
            "The ERP finance module is taking over a minute to load basic reports during month-close.",
            "Every screen in the ERP system is lagging badly since we started month-end closing.",
            "Report generation in the ERP is timing out repeatedly this week, right during closing.",
        ],
        "priority_weights": {"low": 0.0, "medium": 0.25, "high": 0.5, "critical": 0.25},
        "count": 26, "shape": "rising",
    },
    {
        "category": "Email", "domain": "IT Support",
        "subjects": [
            "Email delivery delayed",
            "Outgoing emails bouncing back",
            "Emails not being delivered to external clients",
            "Delay in sending and receiving emails",
            "Client emails keep bouncing with delivery failure",
        ],
        "descriptions": [
            "Emails to external clients are bouncing back with a delivery failure notice.",
            "There's a significant delay of 20-30 minutes before my emails reach recipients.",
            "Several clients have told us our emails aren't arriving at all this week.",
        ],
        "priority_weights": {"low": 0.1, "medium": 0.5, "high": 0.35, "critical": 0.05},
        "count": 29, "shape": "flat",
    },
    {
        "category": "Hardware", "domain": "IT Support",
        "subjects": [
            "Laptop won't boot after Windows update",
            "Laptop stuck on boot screen post-update",
            "System won't start after latest update installed",
            "Blue screen after Windows update, won't boot",
        ],
        "descriptions": [
            "After the latest Windows update installed overnight, my laptop won't get past the boot screen.",
            "Laptop shows a blue screen and restarts in a loop since the forced update last night.",
            "The mandatory update seems to have corrupted something - system won't boot normally now.",
        ],
        "priority_weights": {"low": 0.0, "medium": 0.3, "high": 0.55, "critical": 0.15},
        "count": 22, "shape": "flat",
    },
    {
        "category": "Access", "domain": "IT Support",
        "subjects": [
            "Shared drive access denied",
            "Cannot access shared network drive",
            "Permission denied on shared folder",
            "Lost access to team shared drive",
        ],
        "descriptions": [
            "I get 'access denied' whenever I try to open our team's shared network drive.",
            "Was able to access the shared folder last week, now it says I don't have permission.",
        ],
        "priority_weights": {"low": 0.2, "medium": 0.6, "high": 0.2, "critical": 0.0},
        "count": 20, "shape": "flat",
    },
    {
        "category": "Software", "domain": "Business Apps",
        "subjects": [
            "Software license expired warning",
            "License expiry popup blocking my work",
            "Getting 'license expired' error on startup",
            "Application won't open, license expired message",
        ],
        "descriptions": [
            "I keep getting a license expired popup and can't proceed past it to use the application.",
            "The design software shows 'license expired' on launch even though renewal was confirmed.",
        ],
        "priority_weights": {"low": 0.25, "medium": 0.55, "high": 0.2, "critical": 0.0},
        "count": 18, "shape": "falling",
    },
    {
        "category": "Hardware", "domain": "IT Support",
        "subjects": [
            "Monitor no display after sleep",
            "External monitor blank after laptop wakes up",
            "Screen stays black after waking from sleep mode",
            "Monitor won't wake up, stays black",
        ],
        "descriptions": [
            "After the laptop wakes from sleep, the external monitor stays completely black.",
            "Have to unplug and replug the monitor cable every time the laptop wakes up from sleep.",
        ],
        "priority_weights": {"low": 0.4, "medium": 0.5, "high": 0.1, "critical": 0.0},
        "count": 15, "shape": "falling",
    },
]

# One-off tickets: genuinely distinct topics, each used at most once, so they
# stay as their own size-1 cluster - the realistic "long tail" that should NOT
# get folded into any recurring cluster. Built by pairing an action with a
# topic (not by repeating fixed strings) so the pool is large enough to draw
# from without collisions.
SINGLETON_ACTIONS = [
    "Request for", "Need help with", "Issue with", "Problem reported:",
    "Following up on", "Question about", "Setup required for", "Access needed for",
]
SINGLETON_TOPICS = [
    "a second monitor at my desk", "a laptop for the new joiner in Sales",
    "the quarterly reporting dashboard", "the conference room projector remote",
    "whitelisting a vendor domain in the email filter", "server room badge access",
    "a laptop battery replacement", "an additional software license seat",
    "Bluetooth headset pairing with my laptop", "the newsletter mailing list",
    "desk phone extension 4021 not ringing", "the archived project folder",
    "sticking keyboard keys on desk 4B", "a standing desk conversion",
    "a ticket that was reopened by mistake", "a guest wifi voucher for a vendor",
    "the docking station not charging my laptop", "calendar delegate access",
    "a mouse cursor that freezes intermittently", "webcam detection in the conferencing app",
    "an onboarding checklist for a contractor", "a company laptop asset tag update",
    "expense report software access", "a printer toner cartridge replacement",
    "conference bridge dial-in numbers", "a time-off balance discrepancy",
    "the payroll portal login", "travel booking tool access",
    "an org chart update", "building access card renewal",
    "a second display cable for the meeting room", "the visitor management kiosk",
    "restoring a file from last week's backup", "a shared calendar not syncing",
    "escalation matrix for after-hours support", "the branch office intercom system",
    "a projector bulb replacement", "onboarding a new vendor in the procurement tool",
    "two-factor authentication device re-enrollment", "a desk relocation request",
]


def _unique_singleton_pairs(count: int) -> list[tuple]:
    all_pairs = [(a, t) for a in SINGLETON_ACTIONS for t in SINGLETON_TOPICS]
    random.shuffle(all_pairs)
    return all_pairs[:count]


def _shape_weights(shape: str, days: int) -> list[float]:
    """weights[0] = oldest day, weights[-1] = today."""
    if shape == "rising":
        return [0.4 + 0.9 * (i / (days - 1)) for i in range(days)]
    if shape == "falling":
        return [1.3 - 0.9 * (i / (days - 1)) for i in range(days)]
    return [1.0] * days


def _pick_priority(weights: dict) -> str:
    labels = list(weights.keys())
    probs = list(weights.values())
    return random.choices(labels, weights=probs, k=1)[0]


def _random_datetime_for_day(day_offset_from_start: int) -> datetime:
    day = NOW - timedelta(days=WINDOW_DAYS - 1 - day_offset_from_start)
    return day.replace(
        hour=random.randint(8, 19), minute=random.randint(0, 59), second=random.randint(0, 59), microsecond=0
    )


def generate() -> list[dict]:
    tickets = []
    seq = 1

    for issue in ISSUES:
        weights = _shape_weights(issue["shape"], WINDOW_DAYS)
        day_choices = random.choices(range(WINDOW_DAYS), weights=weights, k=issue["count"])
        for day_offset in day_choices:
            created = _random_datetime_for_day(day_offset)
            status = _pick_status()
            ticket = {
                "_id": f"64f{seq:021d}",
                "ticket_no": f"TCK-2026-{seq:05d}",
                "domain": issue["domain"],
                "subject": random.choice(issue["subjects"]),
                "description": random.choice(issue["descriptions"]),
                "priority": _pick_priority(issue["priority_weights"]),
                "status": status,
                "reporter": {
                    "name": random.choice(REPORTER_NAMES),
                    "department": random.choice(DEPARTMENTS),
                },
                "domain_fields": {
                    "category_type": issue["category"],
                    "branch": random.choice(BRANCHES),
                },
                "created_at": created.isoformat().replace("+00:00", "Z"),
            }
            _maybe_close(ticket, created, status)
            tickets.append(ticket)
            seq += 1

    singleton_count = 250
    for action, topic in _unique_singleton_pairs(singleton_count):
        day_offset = random.randint(0, WINDOW_DAYS - 1)
        created = _random_datetime_for_day(day_offset)
        subject = f"{action} {topic}".capitalize()
        status = _pick_status()
        priority = _pick_priority({"low": 0.4, "medium": 0.45, "high": 0.13, "critical": 0.02})
        ticket = {
            "_id": f"64f{seq:021d}",
            "ticket_no": f"TCK-2026-{seq:05d}",
            "domain": random.choice(["IT Support", "Business Apps", "Facilities"]),
            "subject": subject,
            "description": subject + ". Please assist when possible.",
            "priority": priority,
            "status": status,
            "reporter": {
                "name": random.choice(REPORTER_NAMES),
                "department": random.choice(DEPARTMENTS),
            },
            "domain_fields": {
                "category_type": random.choice(["Access", "Network", "Hardware", "Software", "Email", "Facilities"]),
                "branch": random.choice(BRANCHES),
            },
            "created_at": created.isoformat().replace("+00:00", "Z"),
        }
        _maybe_close(ticket, created, status)
        tickets.append(ticket)
        seq += 1

    random.shuffle(tickets)
    return tickets


if __name__ == "__main__":
    from src.config import settings

    tickets = generate()
    settings.sample_data_path.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    print(f"Wrote {len(tickets)} synthetic tickets to {settings.sample_data_path}")
