# hardcoded download script for site 257 - Bowra-Dry-A
import csv
import json
import requests

BASE_URL = "https://api.acousticobservatory.org/audio_recordings/filter"

COOKIE_VALUE = "_baw_session=%2F0qbuMkrrRvc5uTGm4gMIu3W7ibCuYsdP1a9HAJQzOOr39nMjwfK%2BYTyfukoeATtBIdk2uQwWtY%2BES5EkSXHUq3WlfBGitILZRoezGAa%2FghlkEhqnCcP3Pt3LjE4OOMD1dQQI5r2q3yv7%2FZosGaL%2FwAuI2K1EMQyEYGq6Re9CANRIOkDgHbgfoe3E3mp4P1s87g9QYGcbGoOLaznVGEtgjjEkpjk4J1sgtryOXhQjNSCwY2z2BriGir5BlKebU8qA4%2FBQWrFL1rrLNanGEsDjGfUAps%3D--idFv5qY47j%2F8sGxQ--WPxg0xe6ZkANq9UzNFuhfA%3D%3D"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://data.acousticobservatory.org",
    "Referer": "https://data.acousticobservatory.org/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.4 Safari/605.1.15"
    ),
}

COOKIES = {"_baw_session": COOKIE_VALUE}
SITE_ID = 257

FIELDS = [
    "id",
    "uuid",
    "recorded_date",
    "site_id",
    "duration_seconds",
    "sample_rate_hertz",
    "channels",
    "bit_rate_bps",
    "media_type",
    "data_length_bytes",
    "status",
    "created_at",
    "creator_id",
    "deleted_at",
    "deleter_id",
    "updated_at",
    "updater_id",
    "file_hash",
    "uploader_id",
    "original_file_name",
    "canonical_file_name",
    "recorded_date_timezone",
    "recorded_utc_offset",
]


def payload(page: int) -> dict:
    return {
        "sorting": {"order_by": "recorded_date", "direction": "asc"},
        "filter": {"and": [{"sites.id": {"eq": SITE_ID}}]},
        "paging": {"page": page},
    }


def flatten_row(item: dict) -> dict:
    row = {field: item.get(field) for field in FIELDS}
    row["notes_relative_path"] = (item.get("notes") or {}).get("relative_path")
    return row


with requests.Session() as session:
    first = session.post(BASE_URL, headers=HEADERS, cookies=COOKIES, json=payload(1), timeout=60)
    first.raise_for_status()
    first_json = first.json()

    max_page = first_json["meta"]["paging"]["max_page"]
    all_items = list(first_json.get("data", []))

    for page in range(2, max_page + 1):
        print(f"Fetching page {page}/{max_page}")
        resp = session.post(BASE_URL, headers=HEADERS, cookies=COOKIES, json=payload(page), timeout=60)
        resp.raise_for_status()
        all_items.extend(resp.json().get("data", []))

with open("all_items.json", "w", encoding="utf-8") as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)

with open("all_items.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS + ["notes_relative_path"])
    writer.writeheader()
    for item in all_items:
        writer.writerow(flatten_row(item))

print(f"Done. Wrote {len(all_items)} items to all_items.json and all_items.csv")