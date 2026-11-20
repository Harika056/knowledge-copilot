"""Synthetic sample database for the structured-data path. Replace with your real data source."""
import random
import sqlite3
from pathlib import Path


def build_sample_db(path: str, seed: int = 7) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).unlink(missing_ok=True)
    rnd, con = random.Random(seed), sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, amount REAL, status TEXT);
        CREATE TABLE secrets (id INTEGER PRIMARY KEY, note TEXT);
        INSERT INTO secrets VALUES (1, 'not for the copilot');""")
    regions = ["North", "South", "East", "West"]
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(i, f"Customer {i}", rnd.choice(regions)) for i in range(1, 41)])
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)",
                    [(i, rnd.randint(1, 40), f"2026-{rnd.randint(1, 10):02d}-{rnd.randint(1, 28):02d}",
                      round(rnd.uniform(20, 900), 2), rnd.choice(["PAID", "PAID", "PAID", "REFUNDED", "PENDING"]))
                     for i in range(1, 201)])
    con.commit()
    con.close()
