from src.utils import create_list_waterport_visits_in_between_rwds

waterport_visits = [1, 5, 7, 8, 10, 11]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits)
ref_answer = [[1], [5], [7, 8]]  # note that it does not include the visits after the last reward

print(answer)

assert  answer == ref_answer, \
    "Error! Waterport visits in between reward deliveries %s do not match the reference answer %s" % (answer, ref_answer)

waterport_visits = [1, 5, 7, 8]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits)
ref_answer = [[1], [5], [7, 8]]

print(answer)

assert  answer == ref_answer, \
    "Error! Waterport visits in between reward deliveries %s do not match the reference answer %s when there are " \
    "no waterport visits after the last reward" % (answer, ref_answer)

waterport_visits = [1, 5, 7, 8, 10, 11]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits, include_wp_visits_after_last_rwd=True)
ref_answer = [[1], [5], [7, 8], [10, 11]]  # note that it does not include the visits after the last reward

print(answer)

assert  answer == ref_answer, \
    "Error! Waterport visits in between reward deliveries %s do not match the reference answer %s" % (answer, ref_answer)
