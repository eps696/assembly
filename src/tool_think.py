from tools import bool_, int_, tool

class ThinkingServer:
    def __init__(self):
        self.thoughts = []
        self.branches = {}  # Map of branch_id to list of thoughts in that branch
        self.local_mode = False
    
    def add_thought(self, thought, thought_num=None, total_thoughts=None, next_thought_needed=True,
                    is_revision=False, revises_thought=None, branch_from_thought=None, branch_id=None, needs_more_thoughts=False):
        # Validation
        if is_revision and revises_thought is None:
            raise ValueError("revises_thought must be provided when is_revision is True")
        if branch_from_thought is not None and branch_id is None:
            raise ValueError("branch_id must be provided when branch_from_thought is specified")

        # Auto-numbering: don't trust small LLMs to track sequence correctly
        actual_thought_num = len(self.thoughts) + 1

        # Detect new problem: LLM says thought_num=1 but we already have thoughts
        is_new_problem = (thought_num == 1 and len(self.thoughts) > 0 and not self.local_mode)
        if is_new_problem:
            actual_thought_num = 1  # Reset numbering for new problem context

        # Use LLM's total_thoughts as estimate, but ensure it's at least current position
        if total_thoughts is None or total_thoughts < actual_thought_num:
            total_thoughts = actual_thought_num

        # Auto-detect branch continuation: if previous thought was in a branch, continue it
        current_branch_id = branch_id
        current_branch_from = branch_from_thought
        if current_branch_id is None and not is_revision and len(self.thoughts) > 0:
            prev_thought = self.thoughts[-1]
            if prev_thought.get("branch_id"):
                current_branch_id = prev_thought.get("branch_id")
                current_branch_from = prev_thought.get("branch_from_thought")

        thought_obj = {
            "thought": thought,
            "thought_num": actual_thought_num,
            "thought_num_from_llm": thought_num,  # Keep LLM's value for debugging
            "total_thoughts": total_thoughts,
            "next_thought_needed": next_thought_needed,
            "is_revision": is_revision,
            "revises_thought": revises_thought if is_revision else None,
            "branch_from_thought": current_branch_from if current_branch_id else None,
            "branch_id": current_branch_id,
            "needs_more_thoughts": needs_more_thoughts,
            "is_new_problem": is_new_problem
        }
        self.thoughts.append(thought_obj)

        # Track branch membership
        if current_branch_id:
            if current_branch_id not in self.branches:
                self.branches[current_branch_id] = []
            self.branches[current_branch_id].append(thought_obj)

        return {
            "status": "success",
            "thought": {
                "thought_num": actual_thought_num,
                "total_thoughts": total_thoughts,
                "next_thought_needed": next_thought_needed,
                "branches": list(self.branches.keys()),
                "thought_history_length": len(self.thoughts)
            }
        }
    
    def get_thoughts(self):
        return self.thoughts
    
    def get_branches(self):
        return self.branches
    
    def reset(self):
        self.thoughts = []
        self.branches = {}
        
    def get_local_mode(self):
        return self.local_mode

    def set_local_mode(self, local_mode):
        self.local_mode = local_mode

server = ThinkingServer()

@tool
def sequential_thinking(thought: str, next_thought_needed: bool=True, thought_num: int=None, total_thoughts: int=None,
                        is_revision: bool=False, revises_thought: int=None, branch_from_thought: int=None, branch_id: str=None, needs_more_thoughts: bool=False):
    """Dynamic and reflective problem-solving through thoughts; helping analyze problems through a flexible thinking process that can adapt and evolve, where each thought can build on, question, or revise previous insights as understanding deepens.

    thought: Your current thinking step
    next_thought_needed: Whether another thought step is needed (default: true)
    thought_num: Optional hint for sequence number (auto-managed by the tool)
    total_thoughts: Estimate of thoughts needed (auto-adjusted if too low)
    is_revision: Indicating if this thought revises previous thinking
    revises_thought: If is_revision is true, which thought number is being reconsidered (min: 1)
    branch_from_thought: Branching point thought number (min: 1)
    branch_id: Branch identifier
    needs_more_thoughts: If reaching end but realizing more thoughts needed
    """
    # Parse LLM values with safe defaults - None means "let the tool decide"
    parsed_thought_num = int_(thought_num, None) if thought_num is not None else None
    parsed_total = int_(total_thoughts, None) if total_thoughts is not None else None
    parsed_revises = int_(revises_thought, None) if revises_thought is not None else None
    parsed_branch_from = int_(branch_from_thought, None) if branch_from_thought is not None else None
    parsed_branch_id = None if (branch_id is None or (isinstance(branch_id, str) and 'null' in branch_id.lower())) else branch_id

    result = server.add_thought(
        thought=thought,
        thought_num=parsed_thought_num,
        total_thoughts=parsed_total,
        next_thought_needed=bool_(next_thought_needed),
        is_revision=bool_(is_revision),
        revises_thought=parsed_revises,
        branch_from_thought=parsed_branch_from,
        branch_id=parsed_branch_id,
        needs_more_thoughts=bool_(needs_more_thoughts)
    )
    # Format result with human-readable content for LLM tool response
    return {
        "status": "success",
        "content": f"Thought #{thought_num or 1}: {thought}",
        "thought_details": result['thought']
    }

def get_thoughts():
    return server.get_thoughts()

def get_branches():
    return server.get_branches()

def reset_thinking():
    server.reset()

def set_local_mode(local_mode):
    if server.get_local_mode() != local_mode and local_mode:
        server.reset()
    server.set_local_mode(local_mode)

def print_thoughts(start_index=0, end_index=None, show_branch_info=True):
    thoughts = get_thoughts()
    if end_index is None:
        end_index = len(thoughts)

    selected_thoughts = thoughts[start_index:end_index]
    total_in_range = len(selected_thoughts)
    print(f"\n--- Thinking ({total_in_range} total):")

    branches = get_branches()
    branch_displayed = {}  # Track which thoughts have been displayed with branch info

    for i, t in enumerate(selected_thoughts):
        type_str = ""

        # Mark new problem boundaries
        if t.get("is_new_problem", False):
            type_str = " [NEW PROBLEM]"

        # Show revision info
        if t.get("is_revision", False):
            revised_num = t.get("revises_thought")
            type_str += f" (REVISION of step {revised_num})"
        # Show branch info (but not for revisions)
        elif t.get("branch_id"):
            branch_from = t.get("branch_from_thought")
            branch_id = t.get("branch_id")
            if branch_id not in branch_displayed or branch_displayed[branch_id] == t["thought_num"]:
                type_str += f" ({branch_id}, from step {branch_from})"
                branch_displayed[branch_id] = t["thought_num"] + 1
            else:
                type_str += f" (Branch: {branch_id})"

        # Display with auto-managed thought_num
        print(f"\n- {t['thought_num']}/{total_in_range}{type_str}: {t['thought']}")

    if show_branch_info and branches:
        print("\n--- Branches:")
        for branch_id, branch_thoughts in branches.items():
            thought_nums = [t.get("thought_num") for t in branch_thoughts]
            branch_count = len(branch_thoughts)
            origin = branch_thoughts[0].get("branch_from_thought", "unknown")
            print(f"- '{branch_id}': {branch_count} thoughts, from step {origin}, steps {thought_nums}")

if __name__ == "__main__":
    from tools import get_schemas
    for x in get_schemas('openai'): print(x)
