ps -u | grep "python mp_classify" | awk '{ print  $2 }' | xargs kill
