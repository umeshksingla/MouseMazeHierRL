from src.utils import create_list_waterport_visits_in_between_rwds

# Test include only visits in between rewards, with waterport visits after last reward
waterport_visits = [1, 5, 7, 8, 10, 11]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits)
ref_answer = [[1], [5], [7, 8]]  # note that it does not include the visits after the last reward

print(answer)
assert  answer == ref_answer, \
    "Error! Waterport visits in between reward deliveries %s do not match the reference answer %s" % (answer, ref_answer)

# Test include only visits in between rewards, with no waterport visits after last reward
waterport_visits = [1, 5, 7, 8]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits)
ref_answer = [[1], [5], [7, 8]]

print(answer)
assert  answer == ref_answer, \
    "Error! Waterport visits in between reward deliveries %s do not match the reference answer %s when there are " \
    "no waterport visits after the last reward" % (answer, ref_answer)


# Test include only visits in between rewards, with no waterport visits after last reward, and only one visit in between two last rewards
waterport_visits = [1, 5, 7]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits)
ref_answer = [[1], [5], [7]]

print(answer)
assert  answer == ref_answer, \
    "Error! Waterport visits in between reward deliveries %s do not match the reference answer %s when there are " \
    "no waterport visits after the last reward, and only one visit in between two last rewards" % (answer, ref_answer)


# Test include only visits in between rewards, with only one waterport visit after last reward
waterport_visits = [1, 5, 7, 8, 10]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits, include_wp_visits_after_last_rwd=False)
ref_answer = [[1], [5], [7, 8]]  # note that it does not include the visits after the last reward

print(answer)
assert  answer == ref_answer, \
    "Error! Waterport visits in between (and after) reward deliveries %s do not match the reference answer %s" % (answer, ref_answer)


# Test include visits after last reward true, with waterport visits after last reward
waterport_visits = [1, 5, 7, 8, 10, 11]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits, include_wp_visits_after_last_rwd=True)
ref_answer = [[1], [5], [7, 8], [10, 11]]  # note that it does not include the visits after the last reward

print(answer)
assert  answer == ref_answer, \
    "Error! Waterport visits in between (and after) reward deliveries %s do not match the reference answer %s" % (answer, ref_answer)

# Test include visits after last reward true, with only one waterport visit after last reward
waterport_visits = [1, 5, 7, 8, 10]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits, include_wp_visits_after_last_rwd=True)
ref_answer = [[1], [5], [7, 8], [10]]  # note that it does not include the visits after the last reward

print(answer)
assert  answer == ref_answer, \
    "Error! Waterport visits in between (and after) reward deliveries %s do not match the reference answer %s" % (answer, ref_answer)


# Test function with include visits after last reward true, without waterport visits after last reward
waterport_visits = [1, 5, 7, 8]
rwd_visits = [2, 6, 9]

answer = create_list_waterport_visits_in_between_rwds(waterport_visits, rwd_visits, include_wp_visits_after_last_rwd=True)
ref_answer = [[1], [5], [7, 8]]

print(answer)
assert  answer == ref_answer, \
    "Error! Waterport visits in between (and after) reward deliveries %s do not match the reference answer %s" % (answer, ref_answer)
