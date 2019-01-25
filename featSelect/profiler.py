import pstats

p = pstats.Stats('times.txt')
p.sort_stats('cumtime').print_stats(20)