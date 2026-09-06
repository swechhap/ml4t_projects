""""""  		  	   		 		  		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 		  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 		  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 		  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 		  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 		  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 		  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 		  		  		  		    	 		 		   		 		  
or edited.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 		  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 		  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 		  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Student Name: Swechha Pandey (replace with your name)  		  	   		 		  		  		  		    	 		 		   		 		  
GT User ID: spandey343 (replace with your User ID)  		  	   		 		  		  		  		    	 		 		   		 		  
GT ID: 904262734 (replace with your GT ID)  		  	   		 		  		  		  		    	 		 		   		 		  
"""  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
  		  	   		 		  		  		  		    	 		 		   		 		  
def author():  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 		  		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    return "spandey343"  # replace tb34 with your Georgia Tech username.
  		  	   		 		  		  		  		    	 		 		   		 		  
def study_group():
    return "spandey343"

def gtid():  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 		  		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    return 904262734  # replace with your GT ID number
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 		  		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 		  		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 		  		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    result = False
    if np.random.rand() < win_prob:
        result = True

    return result  		  	   		 		  		  		  		    	 		 		   		 		  

def run_episode(win_prob):
     winnings = 0
     bet = 1

     index = 1
     winnings_array = np.zeros(1001)

     while winnings < 80 and index < 1001:
         result = get_spin_result(win_prob)
         if result:
             winnings = winnings + bet
             bet = 1

         else:
             winnings = winnings - bet
             bet = 2 * bet
         winnings_array[index] = winnings
         index += 1
     winnings_array[index:] = winnings
     return winnings_array


def run_episode_bankroll(win_prob):
    winnings = 0
    bet = 1

    index = 1
    winnings_array = np.zeros(1001)

    while winnings < 80 and index < 1001 and winnings > -256:
        result = get_spin_result(win_prob)
        if bet > 256 + winnings:
            actual_bet = 256 + winnings
        else:
            actual_bet = bet
        if result:
            winnings = winnings + actual_bet
            bet = 1

        else:

            winnings = winnings - actual_bet
            bet = 2 * bet

        winnings_array[index] = winnings
        index += 1
    winnings_array[index:] = winnings
    return winnings_array

def make_plot(center, std, center_label, title, filename):
    plt.figure()
    plt.plot(center, label=center_label)
    plt.plot(center + std, label=f'{center_label} + Std Dev')
    plt.plot(center - std, label=f'{center_label} - Std Dev')
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.title(title)
    plt.xlabel('Spin Number')
    plt.ylabel('Winnings ($)')
    plt.legend(loc='lower left')
    plt.savefig(filename)
    plt.close()


def test_code():
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    win_prob = 18/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    # add your code here to implement the experiments  		  	   		 		  		  		  		    	 		 		   		 		  
    exp1_matrix = np.zeros((10, 1001))
    for i in range(10):
        exp1_matrix[i] = run_episode(win_prob)
    plt.figure()
    for i in range(10):
        plt.plot(exp1_matrix[i], label=f'Episode {i+1}')
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.title('Figure 1: 10 Episodes of Martingale Betting')
    plt.xlabel('Spin number')
    plt.ylabel('Winnings ($)')
    plt.legend(loc='lower left')
    plt.savefig("figure1.png")
    plt.close()

    win_matrix = np.zeros((1000, 1001))
    for i in range(1000):
        win_matrix[i] = run_episode(win_prob)
    mean_line = np.mean(win_matrix, axis=0)
    std_line = np.std(win_matrix, axis=0)
    print(np.mean(win_matrix[:, -1]))
    make_plot(mean_line, std_line,
              'Mean',
              'Figure 2: Mean Winnings Over 1000 Episodes',
              'figure2.png')


    median_line = np.median(win_matrix, axis=0)
    make_plot(median_line, std_line,
              'Median',
              'Figure 3: Median Winnings Over 1000 Episodes',
              'figure3.png')
    # plot = plt.make_plot(win_prob, labels, filename)

    exp2_matrix = np.zeros((1000, 1001))
    for i in range(1000):
        exp2_matrix[i] = run_episode_bankroll(win_prob)
    mean_line = np.mean(exp2_matrix, axis=0)
    std_line = np.std(exp2_matrix, axis=0)
    make_plot(mean_line, std_line,
              'Mean',
              'Figure 4: Mean Winnings with $256 Bankroll (1000 Episodes)',
              'figure4.png')
    final_winnings_exp2 = exp2_matrix[:, -1]

    number_reaching_80_exp2 = np.sum(final_winnings_exp2 == 80)
    probability_80_exp2 = number_reaching_80_exp2 / 1000

    print("Experiment 2")
    print("Episodes reaching $80:", number_reaching_80_exp2)
    print("Estimated probability:", probability_80_exp2)
    print("Estimated probability (%):", probability_80_exp2 * 100)
    median_line = np.median(exp2_matrix, axis=0)
    expected_value_exp2 = np.mean(final_winnings_exp2)


    print("Estimated expected value:", expected_value_exp2)
    make_plot(median_line, std_line,
              'Median',
              'Figure 5: Median Winnings with $256 bankroll Over 1000 Episodes',
              'figure5.png')
  		  	   		 		  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 		  		  		  		    	 		 		   		 		  
    test_code()  		  	   		 		  		  		  		    	 		 		   		 		  
