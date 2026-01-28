# ==================== Imports ====================
from __future__ import annotations
import time
from typing import Any, Iterable
import hivemind  # type: ignore


# ==================== Peer Synchronization ====================


# Wait until all num_peers_target amongst num_peers have set a DHT flag for a specific prefix which optimally includes the iteration number
def wait_for_all(
    dht: Any,
    prefix: str,
    num_peers: int,
    num_peers_target: int,
    delay: float = 0.05,
    max_delays: int = 200,
) -> list[int]:
    delay_counter = 0
    while True:
        ready_peers = 0
        written_members = []
        for peer in range(num_peers):
            key = f"{prefix}_{peer}"
            if dht.get(key) is not None:
                ready_peers += 1
                written_members.append(peer)
        if ready_peers == num_peers_target or delay_counter == max_delays:
            return written_members
        time.sleep(delay)
        delay_counter += 1


# Wait until all specified peers have set a DHT flag for a specific prefix which optimally includes the iteration number
def wait_for_peers(
    dht: Any,
    prefix: str,
    peers: Iterable[int],
    delay: float = 0.05,
    max_delays: int = 200,
) -> bool:
    delay_counter = 0
    while True:
        all_set = all(dht.get(f"{prefix}_{peer}", latest=True) is not None for peer in peers)
        if all_set:
            return True
        if delay_counter == max_delays:
            return False
        time.sleep(delay)
        delay_counter += 1


# Wait until all num_peers_target amongst num_peers have set a DHT flag and get all respective stored values
def wait_for_all_and_get(
    dht: Any,
    prefix: str,
    num_peers: int,
    num_peers_target: int,
    delay: float = 0.05,
    max_delays: int = 200,
) -> tuple[list[int], list[Any]]:
    delay_counter = 0
    while True:
        ready_peers = 0
        written_members = []
        received_values = []
        for peer in range(num_peers):
            key = f"{prefix}_{peer}"
            value = dht.get(key)
            if value is not None:
                ready_peers += 1
                written_members.append(peer)
                received_values.append(value.value)
        if ready_peers == num_peers_target or delay_counter == max_delays:
            return written_members, received_values
        time.sleep(delay)
        delay_counter += 1


# ==================== Peer Participation ====================


# Announce participation and count number of participating peers
def count_num_participating_peers(
    peer_id: int,
    iteration: int,
    num_peers: int,
    dht: Any,
    wait_time: float = 5,
) -> tuple[int, list[int]]:
    participation_indicator_key = f"pik_{iteration}"
    dht.store(
        f"{participation_indicator_key}_{peer_id}",
        True,
        expiration_time=hivemind.get_dht_time() + 60,
    )
    time.sleep(wait_time)
    participating_peers = []
    for peer in range(num_peers):
        key = f"{participation_indicator_key}_{peer}"
        if dht.get(key) is not None:
            participating_peers.append(peer)
    return len(participating_peers), sorted(participating_peers)
