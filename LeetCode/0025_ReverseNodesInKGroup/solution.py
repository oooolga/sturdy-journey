# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def __init__(self):
        self.inv_tail = None
        self.ret_head = None
        self.total_node_count, self.c_k = 0, 0
    
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
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
    
def print_ll(head):
    while head:
        print(head.val, end=' ')
        head = head.next
    print()
            
if __name__ == '__main__':
    s = Solution()
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    k = 3
    print_ll(s.reverseKGroup(head, k))
    