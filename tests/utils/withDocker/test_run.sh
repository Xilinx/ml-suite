bash -x build_lib.sh
ssh xsjfislx18 "docker exec anup_container /bin/sh "navigate.sh""

# trigger mail
./mail.sh
