import libs_env.blackbox.black_box_match



match = libs_env.blackbox.black_box_match.BlackBoxMatch("black_box_match_net2.json")


match.run()
match.print_score()
match.save_score()
