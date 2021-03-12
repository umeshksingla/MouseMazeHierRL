from src.utils import create_list_waterport_visits_in_between_rwds

waterport_visits = [1, 5, 7, 8, 10, 11]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits)
ref_answer = [[1], [5], [7, 8], [10, 11]]

print(answer)

assert  answer == ref_answer, \
    "Error! Waterport visits in between reward deliveries do not match the reference answer"
