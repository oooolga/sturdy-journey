# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head):
        hare, tortoise = head, head
        try:
            while True:
                hare = hare.next.next
                tortoise = tortoise.next
                if hare == tortoise:
                    tortoise2 = head
                    while tortoise2 != tortoise:
                        tortoise = tortoise.next
                        tortoise2 = tortoise2.next
                    return tortoise
        except:
            return None

if __name__ == "__main__":
    sol = Solution()
    head = ListNode(3)
    head.next = ListNode(2)
    head.next.next = ListNode(0)
    head.next.next.next = ListNode(-4)
    head.next.next.next.next = head.next
    print(sol.detectCycle(head).val)  # 2