# [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

Given the`head` of a linked list, reverse the nodes of the list `k` at a time, and return the modified list.

`k` is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of `k` then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

## Solution
*Runtime: O(n)*

*Space: O(1)*

We perform a complete traversal of the linked list to calculate `n`, the total number of nodes within the list.

After the first pass, we go through the linked list again, moving one node at a time, to make a reversed linked list. Each time we reach a new node, called `node_i`, we put it at the start of the reversed list using the `add_to_invert` function. Whenever we add a new node to this reversed list, we also keep track of how many nodes are in it and make a note of the last node in the list.

If our reversed list gets to have `k` nodes in it, we add it to the final reversed linked list we've been making. We connect the end of the final linked list we made before to the start of the new reversed list. We use a function called `start_new_invert` to do this.

Finally, every time we're about to make a reversed list of `k` nodes, we first see how many nodes are left in the input linked list. If there are fewer than `k` nodes left, we don't make a new reversed list. Instead, we just add the remaining nodes as they are to the end of our big reversed list.

## Code
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def __init__(self):
        self.inv_tail = None
        self.ret_head = None
        self.total_node_count, self.c_k = 0, 0
    
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        n = 1
        head_copy = head
        while head_copy.next:
            n += 1
            head_copy = head_copy.next

        def add_to_invert(node):
            if self.inv_tail is None:
                self.inv_tail, self.inv_head = node, node
            else:
                node.next = self.inv_head
                self.inv_head = node
            self.c_k += 1
        
        def start_new_invert(next_node):
            if self.ret_head is None:
                self.ret_head = self.inv_head
            else:
                self.last_inv_tail.next = self.inv_head
            self.last_inv_tail = self.inv_tail
            self.inv_tail, self.c_k = None, 0
            self.total_node_count += k
        

        while head:
            next_node = head.next
            add_to_invert(head)
            if self.c_k == k:
                start_new_invert(next_node)
            if n-self.total_node_count < k:
                self.last_inv_tail.next = next_node
                break
            head = next_node
        return self.ret_head
```