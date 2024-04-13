class Solution:
    def judgePoint24(self, cards: list[int]) -> bool:
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

if __name__ == '__main__':
    sol = Solution()
    print(sol.judgePoint24([4,1,8,7])) # True
    print(sol.judgePoint24([1,2,1,2])) # False
    print(sol.judgePoint24([1,5,5,5])) # True
    print(sol.judgePoint24([3,3,8,8])) # True