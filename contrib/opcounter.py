import re
import subprocess
import json
import itertools
import matplotlib.pyplot as plt


def calculate_message_counts(npm):
    process = subprocess.Popen("dualtreecalculator",
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    for (n,p,m) in npm:
        process.stdin.write(f"{n} {p} {m}\n".encode("utf-8"))
    
    process.stdin.close()

    result = []
    for _ in npm:
        content = ""
        while (line := process.stdout.readline()) != b'\n':
            content += line.decode("utf-8")

        result.append(json.loads(content.strip()))

    process.wait()

    return result

# src: https://stackoverflow.com/a/1884277
def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def compute_critical_opcount(messages):
    for rank_info in messages:
        # The first few pushes are from local computations.
        # We are interested in the first push that causes an MPI_Wait
        skip = len(rank_info['local_elements'])
        start = find_nth(rank_info['ops'], 'p', skip + 1)
        rank_info['critical_ops'] = rank_info['ops'][start:]

    return messages


message_counts = list(map(compute_critical_opcount, calculate_message_counts([(400,4,2)])))
print(message_counts)
