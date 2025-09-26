import os

from storage.sqlite_store import init_db, list_recent

DB_PATH = os.getenv("SME_DB_PATH", "./data/signal.db")
init_db(DB_PATH)


def main():
    rows = list_recent(DB_PATH, limit=10)
    if not rows:
        print("No events.")
        return
    print(f"{'id':>4}  {'time':<24} {'user':<10} {'agent':<6} {'sig':<10} {'drift':<5}  ESC")
    for r in rows:
        mark = "!!" if r["escalate_flag"] else "  "
        print(
            f"{r['id']:>4}  {r['timestamp']:<24} {str(r['user_id'])[:10]:<10} {str(r['agent_id'])[:6]:<6} "
            f"{str(r['signal_type'])[:10]:<10} {r['drift_score']!s:<5}  {mark}"
        )


if __name__ == "__main__":
    main()
