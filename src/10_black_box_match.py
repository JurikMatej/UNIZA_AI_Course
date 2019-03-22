import libs.libs_env.blackbox.black_box_match



match = libs.libs_env.blackbox.black_box_match.BlackBoxMatch("black_box_match_config_final.json")


match.run()
match.print_score()
match.save_score()
