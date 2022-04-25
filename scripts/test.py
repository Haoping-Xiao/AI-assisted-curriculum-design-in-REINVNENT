from human import first_curriculum, next_curriculum
import os, time, unittest

class TestExecuteCurriculum(unittest.TestCase):

    # def test_first_curriculum(self):
    #     output_dir=first_curriculum("drd2_activity_1")
    #     time.sleep(10)
    #     self.assertTrue(os.path.exists(os.path.join(output_dir,"results_0")))
    def test_next_curriculum(self):
        output_dir=next_curriculum("/scratch/work/xiaoh2/Thesis/results/curriculum_drd2_activity_1","qed")
        time.sleep(10)
        self.assertTrue(os.path.exists(os.path.join(output_dir,"results_0")))




if __name__=="__main__":
  unittest.main()