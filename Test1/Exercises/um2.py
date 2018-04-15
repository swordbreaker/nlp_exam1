from standard_deviation import *
import scipy.stats as stats

# Aufgabe 6
std = StandardDeviation(20.15, 0.12)

mu = 20.15
sigma = 0.12

#a
std.cdf(mu + 0.03) - std.cdf(mu - 0.03)
#b
1 - std.cdf(20)
#c
std.cdf(19.7)

# Aufgabe 7
avg, std = StandardDeviation.calc_avg_std(0.05, 39.5, 0.95, 48.5)
std_w = StandardDeviation(avg, std)

avg, std = StandardDeviation.calc_avg_std(0.05, 44, 0.95, 52.5)
std_m = StandardDeviation(avg, std)

#a
std_w.cdf(48)
std_m.cdf(48)

#b
std_w.pff(0.9)
std_w.pff(0.99)

#c
std_m.pff(0.9)
std_m.pff(0.99)


# Aufgabe 8
mu = 72
sigma = 15
std = StandardDeviation(72, 15)
#a
std.calc_z(60)
std.calc_z(93)
std.calc_z(72)

#b
# z = (x - mu) / sigma
# z * sigma = x - mu
# z * sigma + mu = x
-1 * sigma + mu
1.6 * sigma + mu
2 * sigma + mu

1 - std.cdf(100)
std.cdf(50)

std.ppf(1-0.6)

# Aufgabe 9
std = StandardDeviation(151, 15)

#a
(std.cdf(155) - std.cdf(120)) * 500

#b
(1 - std.cdf(185)) * 500

# Aufgabe 10
std = StandardDeviation(100, 15)

#a
std.cdf(130) - std.cdf(100)

#b
1-std.cdf(130)

#c
std.ppf(0.9)

# Aufgabe 11

# 62 => 32.31 %
# 69 => 16.15 %

StandardDeviation.calc_avg_std(0.3231, 62, 1 - 0.1615, 69)