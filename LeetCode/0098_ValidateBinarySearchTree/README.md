# [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/description/)

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.


## Solution
Perform a Depth-First Search (DFS) on the Binary Search Tree (BST). For each part of the tree, figure out the smallest and largest values it contains.

Make sure that each main tree node's value is bigger than the largest value on its left subtree but smaller than the smallest value on its right subtree. If not, something's wrong, and you should stop and report an error.

If everything looks good, note down the smallest value of the left subtree and largest value of right subtree to keep track of the range of values as you go.

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        
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
```