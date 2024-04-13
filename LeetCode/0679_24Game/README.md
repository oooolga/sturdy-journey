# [24 Game](https://leetcode.com/problems/24-game/description/)

You are given an integer array `cards` of length `4`. You have four cards, each containing a number in the range `[1, 9]`. You should arrange the numbers on these cards in a mathematical expression using the operators `['+', '-', '*', '/']` and the parentheses `'('` and `')'` to get the value 24.

You are restricted with the following rules:

- The division operator `'/'` represents real division, not integer division.
    * For example, `4 / (1 - 2 / 3) = 4 / (1 / 3) = 12`.
- Every operation done is between two numbers. In particular, we cannot use `'-'` as a unary operator.
    * For example, if `cards = [1, 1, 1, 1]`, the expression `"-1 - 1 - 1 - 1"` is not allowed.
- You cannot concatenate numbers together
    * For example, if `cards = [1, 2, 1, 2]`, the expression `"12 + 12"` is not valid.

Return `true` if you can get such expression that evaluates to `24`, and `false` otherwise.

## Solution (Beats 93%)
In any solution to the problem, there must be at least one—and no more than two—operations that involve combining two cards using a mathematical operator. For example, in the expression `(4/(1-2/3)`, there is a single operation involving the numbers `2` and `3`. Conversely, in the expression `(8-4)*(7-1)`, there are two such operations, one between `8` and `4`, and another between `7` and `1`.

First, we calculate all possible results that can be derived from combining any two cards using mathematical operations, and store these results in the `two_cards` dictionary. The `key` in this dictionary is a tuple of the two card values, and the `value` is a set containing all possible outcomes of the operations between them.

Suppose we select two cards corresponding to any key `(card1, card2)` in the `two_cards` dictionary. We then perform three checks using any combination from `two_cards`:

1. We choose one of the remaining two cards and compute all possible results by combining it with the outcomes from the initial two-card operations. Subsequently, we use the last remaining card to calculate all potential results by integrating it with the outcomes from the previous steps. If any of these results equal `24`, we return `true`.

2. We reverse the order in which we select the last two cards and replicate the first check. This step ensures that all possible orderings of how the cards are used are explored.

3. We refer to the two_cards lookup table to find all possible outcomes of the last two remaining cards. We then compute all possible results from combining the outcomes of the initial two-card operations with these new results to see if any yield `24`. *(TODO: We could enhance the efficiency of the algorithm by adding a check to this step. Specifically, we can verify if we have already evaluated the outcomes of `(A, B)` combined with `(C, D)`. If this comparison has been completed, there's no need to repeat the process for `(C, D)` outcomes with `(A, B)`. This avoids redundant calculations and speeds up the execution of the algorithm.)*

## Code
```
class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        two_cards = {}

        import itertools
        def get_two_results(v, card, if_round=False):
            ret = set((v+card, v-card, card-v, v*card,))
            if v!= 0: ret.add(round(card/v, 3) if if_round else card/v)
            if card!=0: ret.add(round(v/card, 3) if if_round else v/card)
            return ret
        
        def get_set_and_card(s, card, if_check=False):
            ret = set()
            for v in s:
                if not if_check:
                    ret |= get_two_results(v, card)
                else:
                    if 24 in get_two_results(v, card, if_round=True):
                        return True
            return ret if not if_check else False
        
        def check_set_and_set(s1, s2):
            for v1 in s1:
                for v2 in s2:
                    if 24 in get_two_results(v1, v2, if_round=True):
                        return True
            return False

        
        for card1, card2, card3, card4 in itertools.permutations(cards, 4):
            if card1 <= card2:
                two_cards[(card1,card2)] = {"remaining_cards": (card3, card4)}
                two_cards[(card1,card2)]["results"] = get_two_results(card1, card2)
        
        for combo in two_cards.keys():
            
            card3, card4 = two_cards[combo]["remaining_cards"]
            if get_set_and_card(get_set_and_card(two_cards[combo]["results"], card3), card4, if_check=True):
                return True
            if card3 != card4:
                if get_set_and_card(get_set_and_card(two_cards[combo]["results"], card4), card3, if_check=True):
                    return True
            if check_set_and_set(two_cards[combo]["results"], two_cards[(min(card3, card4), max(card3, card4))]["results"]):
                return True
        return False

```