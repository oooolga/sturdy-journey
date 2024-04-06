class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root) -> bool:
        
        def helper(root):
            if root.left:  
                left_lower, left_upper = helper(root.left)
                assert root.val > left_upper
            else:
                left_lower = root.val
            if root.right:
                right_lower, right_upper= helper(root.right)
                assert root.val < right_lower
            else:
                right_upper = root.val
            return left_lower, right_upper
        
        try:
            helper(root)
            return True
        except:
            return False

if __name__ == "__main__":
    testcase1 = {"root": TreeNode(2, TreeNode(1), TreeNode(3))}
    sol = Solution()
    print(sol.isValidBST(**testcase1))
    testcase2 = {"root": TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))}
    print(sol.isValidBST(**testcase2))
    testcase3 = {"root": TreeNode(5, TreeNode(4), TreeNode(6, TreeNode(3), TreeNode(7)))}
    print(sol.isValidBST(**testcase3))
    testcase4 = {"root": TreeNode(2, TreeNode(2), TreeNode(2))}
    print(sol.isValidBST(**testcase4))
    testcase5 = {"root": TreeNode(5, TreeNode(1), TreeNode(7, TreeNode(6), TreeNode(8)))}
    print(sol.isValidBST(**testcase5))