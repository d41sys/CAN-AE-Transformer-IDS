import time
import sys

class ACNode:
    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.is_end_of_pattern = False  # Flag to indicate end of a pattern
        self.failure = None  # Failure transition
        self.output = []  # Output labels


class AhoCorasick:
    def __init__(self):
        self.root = ACNode()  # Root of the trie

    def add_pattern(self, pattern):
        node = self.root
        for char in pattern:
            if char not in node.children:
                node.children[char] = ACNode()
            node = node.children[char]
        node.is_end_of_pattern = True
        node.output.append(pattern)

    def build_failure_transitions(self):
        queue = []

        # Initialize failure transitions of depth-1 nodes
        for child in self.root.children.values():
            child.failure = self.root
            queue.append(child)

        # Build failure transitions using BFS
        while queue:
            current = queue.pop(0)

            for char, child in current.children.items():
                queue.append(child)
                failure_state = current.failure

                while failure_state != self.root and char not in failure_state.children:
                    failure_state = failure_state.failure

                if char in failure_state.children:
                    child.failure = failure_state.children[char]
                else:
                    child.failure = self.root

                # Propagate output labels
                child.output += child.failure.output

    def match_patterns(self, text):
        current = self.root
        matches = []

        for char in text:
            while current != self.root and char not in current.children:
                current = current.failure

            if char in current.children:
                current = current.children[char]
            else:
                current = self.root

            # Output matching patterns
            if current.output:
                matches.extend(current.output)

        return matches


# Example usage
rule_based = {'0002': 8, '00a0': 8, '00a1': 8, '0105': 6, '0130': 8, '0131': 8, '0140': 8, '0153': 8, '018f': 8, '01f1': 8, '0260': 8, '02a0': 8, '02b0': 5, '02c0': 8, '0316': 8, '0329': 8, '0350': 8, '0370': 8, '0430': 8, '043f': 8, '0440': 8, '04b1': 8, '04f0': 8, '0545': 8, '05a0': 8, '05a2': 8, '05f0': 2, '0608': 8, '0690': 8, '06a1': 8, '06aa': 5, '071b': 8, '07cf': 8, '07d9': 8, '07da': 8, '07de': 8, '07e8': 8, '07e9': 8}



input_data = "07dse"

ac = AhoCorasick()

# Add rules to the Aho-Corasick trie
for rule, dlc in rule_based.items():
    ac.add_pattern(rule)

# Build failure transitions
ac.build_failure_transitions()

start_time = time.time()

# Match the input data against the rules
matches = ac.match_patterns(input_data)
if matches:
    print("Input:", input_data, "Matches:", matches)
else:
    print("Input:", input_data, "No match")

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Measure memory usage
memory_usage = sys.getsizeof(ac)
print("Memory usage:", memory_usage, "bytes")