#!/usr/bin/env python

import unittest
import os
import sys
import shutil

np_installed = True

try:
    import libNeuroProofRag as neuroproof
except ImportError:
    np_installed = False

class testPackages(unittest.TestCase):
    def testRaveler(self):
        self.assertTrue(os.path.exists("/usr/local/raveler-hdf"))
    
    def testSyngeo(self):
        syngeo_installed = True
        try:
            import syngeo
        except ImportError:
            syngeo_installed = False
        self.assertTrue(syngeo_installed)   
 
    def testVigra(self):
        vigra_installed = True
        try:
            import vigra 
        except ImportError:
            vigra_installed = False
        self.assertTrue(vigra_installed)

    def testNeuroProof(self):
        self.assertTrue(np_installed)

    def testIlastik(self):
        found_exe = False
        for dir in os.getenv("PATH").split(':'):                                           
            if (os.path.exists(os.path.join(dir, "ilastik_batch_fast"))):
                found_exe = True
                break
        self.assertTrue(found_exe)

        
class testModules(unittest.TestCase):
    def gen_watershed(self):
        from ray import imio
        import numpy
        from skimage import morphology as skmorph
        from scipy.ndimage import label

        self.datadir = os.path.abspath(os.path.dirname(sys.modules["ray"].__file__)) + "/testdata/"

        prediction = imio.read_image_stack(self.datadir +"pixelprobs.h5",
                group='/volume/prediction', single_channel=False)
        
        boundary = prediction[...,0]
        seeds = label(boundary==0)[0]
        supervoxels = skmorph.watershed(boundary, seeds)
        return supervoxels, boundary, prediction

    def testNPRFBuild(self):
        if not np_installed:
            return
        from ray import stack_np
        from ray import classify
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["ray"].__file__)) + "/testdata/"

        cl = classify.RandomForest()
        fm_info = cl.load_from_disk(self.datadir + "agglomclassifier_np.rf.h5")

        watershed, boundary, prediction = self.gen_watershed()
        stack = stack_np.Stack(watershed, prediction, single_channel=False,
                classifier=cl, feature_info=fm_info)
        self.assertEqual(stack.number_of_nodes(), 3629)
        stack.agglomerate(0.1)
        self.assertEqual(stack.number_of_nodes(), 83)
        stack.remove_inclusions()
        self.assertEqual(stack.number_of_nodes(), 83)

    def testAggloRFBuild(self):
        from ray import agglo
        from ray import features
        from ray import classify
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["ray"].__file__)) + "/testdata/"

        cl = classify.RandomForest()
        fm_info = cl.load_from_disk(self.datadir + "agglomclassifier.rf.h5")
        fm = features.io.create_fm(fm_info)
        mpf = agglo.classifier_probability(fm, cl)

        watershed, dummy, prediction = self.gen_watershed()
        stack = agglo.Rag(watershed, prediction, mpf, feature_manager=fm, nozeros=True)
        self.assertEqual(stack.number_of_nodes(), 3630)
        stack.agglomerate(0.1)
        self.assertEqual(stack.number_of_nodes(), 88)
        stack.remove_inclusions()
        self.assertEqual(stack.number_of_nodes(), 86)

    def testNPBuild(self):
        if not np_installed:
            return
        from ray import stack_np
        watershed, boundary, dummy = self.gen_watershed()
        stack = stack_np.Stack(watershed, boundary)
        self.assertEqual(stack.number_of_nodes(), 3629)
        stack.agglomerate(0.5)
        self.assertEqual(stack.number_of_nodes(), 86)
        stack.remove_inclusions()
        self.assertEqual(stack.number_of_nodes(), 84)

    def testAggoBuild(self):
        from ray import agglo
        watershed, boundary, dummy = self.gen_watershed()
        stack = agglo.Rag(watershed, boundary, nozeros=True)
        self.assertEqual(stack.number_of_nodes(), 3630)
        stack.agglomerate(0.5)
        self.assertEqual(stack.number_of_nodes(), 61)
        stack.remove_inclusions()
        self.assertEqual(stack.number_of_nodes(), 61)

    def testWatershed(self):
        self.gen_watershed()


class testFlows(unittest.TestCase):
    def testNPFlow(self):
        import ray
        if not np_installed:
            return
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["ray"].__file__)) + "/testdata"
        writedir = "/tmp/NPregtest"
        
        if os.path.exists(writedir):
            shutil.rmtree(writedir)
        os.makedirs(writedir)
        
    
        configfile = open(self.datadir + "/config.json", 'r')
        configstr = configfile.read()
        configstr = configstr.replace("ZZ", self.datadir)
        writefile = open("/tmp/NPregtest/config.json", 'w')
        writefile.write(configstr) 
        writefile.close()

        os.system("ray-segmentation-pipeline " + writedir +  " --config-file " +
               "/tmp/NPregtest/config.json --regression --enable-use-neuroproof >& /dev/null")
        
        self.assertEqual(open(self.datadir + "/seg-pipeline-np.log", 'r').read(),
                        open("/tmp/NPregtest/.seg-pipeline.log", 'r').read())         

    def testRegFlow(self):
        import ray
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["ray"].__file__)) + "/testdata"
        writedir = "/tmp/regtest"
        
        if os.path.exists(writedir):
            shutil.rmtree(writedir)
        os.makedirs(writedir)
       
        configfile = open(self.datadir + "/config.json", 'r')
        configstr = configfile.read()
        configstr = configstr.replace("ZZ", self.datadir)
        writefile = open("/tmp/regtest/config.json", 'w')
        writefile.write(configstr) 
        writefile.close()

        os.system("ray-segmentation-pipeline " + writedir +  " --config-file " +
                "/tmp/regtest/config.json --regression --disable-use-neuroproof >& /dev/null")
        
        self.assertEqual(open(self.datadir + "/seg-pipeline.log", 'r').read(),
                        open("/tmp/regtest/.seg-pipeline.log", 'r').read())         


def entrypoint(argv):
    suite1 = unittest.TestLoader().loadTestsFromTestCase(testPackages)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(testModules)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(testFlows)
    suite_combo =  unittest.TestSuite([suite1, suite2, suite3])
    unittest.TextTestRunner(verbosity=2).run(suite_combo)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
