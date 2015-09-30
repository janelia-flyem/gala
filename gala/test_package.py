#!/usr/bin/env python

import unittest
import os
import sys
import shutil
import json

np_installed = True

try:
    import libNeuroProofRag as neuroproof
except ImportError:
    np_installed = False

class testPackages(unittest.TestCase):
    def testRaveler(self):
        found_exe = False
        for dir in os.getenv("PATH").split(':'):                                           
            if (os.path.exists(os.path.join(dir, "compilestack"))):
                found_exe = True
                break
        self.assertTrue(found_exe)
    
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
            if (os.path.exists(os.path.join(dir, "ilastik_headless"))):
                found_exe = True
                break
        self.assertTrue(found_exe)

        
class testModules(unittest.TestCase):
    def gen_watershed(self):
        from gala import imio
        import numpy
        from skimage import morphology as skmorph
        from scipy.ndimage import label

        self.datadir = os.path.abspath(os.path.dirname(sys.modules["gala"].__file__)) + "/testdata/"

        prediction = imio.read_image_stack(self.datadir +"pixelprobs.h5",
                group='/volume/prediction', single_channel=False)
        
        boundary = prediction[...,0]
        seeds = label(boundary==0)[0]
        supervoxels = skmorph.watershed(boundary, seeds)
        return supervoxels, boundary, prediction

    def testNPRFBuild(self):
        if not np_installed:
            self.assertTrue(np_installed)
        from gala import stack_np
        from gala import classify
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["gala"].__file__)) + "/testdata/"

        cl = classify.load_classifier(self.datadir + "agglomclassifier_np.rf.h5")
        fm_info = json.loads(str(cl.feature_description))

        watershed, boundary, prediction = self.gen_watershed()
        stack = stack_np.Stack(watershed, prediction, single_channel=False,
                classifier=cl, feature_info=fm_info)
        self.assertEqual(stack.number_of_nodes(), 3629)
        stack.agglomerate(0.1)
        self.assertEqual(stack.number_of_nodes(), 80)
        stack.remove_inclusions()
        self.assertEqual(stack.number_of_nodes(), 78)

    def testAggloRFBuild(self):
        from gala import agglo
        from gala import features
        from gala import classify
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["gala"].__file__)) + "/testdata/"

        cl = classify.load_classifier(self.datadir + "agglomclassifier.rf.h5")
        fm_info = json.loads(str(cl.feature_description))
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
            self.assertTrue(np_installed)
        from gala import stack_np
        watershed, boundary, dummy = self.gen_watershed()
        stack = stack_np.Stack(watershed, boundary)
        self.assertEqual(stack.number_of_nodes(), 3629)
        stack.agglomerate(0.5)
        self.assertEqual(stack.number_of_nodes(), 82)
        stack.remove_inclusions()
        self.assertEqual(stack.number_of_nodes(), 82)

    def testAggoBuild(self):
        from gala import agglo
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
        import gala
        if not np_installed:
            self.assertTrue(np_installed)
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["gala"].__file__)) + "/testdata"
        writedir = "/tmp/NPregtest"
        
        if os.path.exists(writedir):
            #shutil.rmtree(writedir)
            os.system("rm -f " + writedir + "/*")
            os.system("rm -f " + writedir + "/.??*")
        else:
	    os.makedirs(writedir)
        
    
        configfile = open(self.datadir + "/config.json", 'r')
        configstr = configfile.read()
        configstr = configstr.replace("ZZ", self.datadir)
        writefile = open("/tmp/NPregtest/config.json", 'w')
        writefile.write(configstr) 
        writefile.close()

        os.system("gala-segmentation-pipeline " + writedir +  " --config-file " +
               "/tmp/NPregtest/config.json --regression --enable-use-neuroproof >& /dev/null")
        
        log_data = open(self.datadir + "/seg-pipeline-np.log", 'r').read()
	log_data = log_data.replace("ZZ", self.datadir)

        os.system("chmod 777 " + writedir + "/*")
        os.system("chmod 777 " + writedir + "/.??*")
	
        self.assertEqual(log_data,
                        open("/tmp/NPregtest/.seg-pipeline.log", 'r').read())         

    def testRegFlow(self):
        import gala
        self.datadir = os.path.abspath(os.path.dirname(sys.modules["gala"].__file__)) + "/testdata"
        writedir = "/tmp/regtest"
        
        if os.path.exists(writedir):
            #shutil.rmtree(writedir)
            os.system("rm -f " + writedir + "/*")
            os.system("rm -f " + writedir + "/.??*")
        else:
	    os.makedirs(writedir)
       
        configfile = open(self.datadir + "/config.json", 'r')
        configstr = configfile.read()
        configstr = configstr.replace("ZZ", self.datadir)
        writefile = open("/tmp/regtest/config.json", 'w')
        writefile.write(configstr) 
        writefile.close()

        os.system("gala-segmentation-pipeline " + writedir +  " --config-file " +
                "/tmp/regtest/config.json --regression --disable-use-neuroproof >& /dev/null")
        
        log_data = open(self.datadir + "/seg-pipeline.log", 'r').read()
	log_data = log_data.replace("ZZ", self.datadir)
        
        os.system("chmod 777 " + writedir + "/*")
        os.system("chmod 777 " + writedir + "/.??*")
        
        self.assertEqual(log_data,
                        open("/tmp/regtest/.seg-pipeline.log", 'r').read())         


def entrypoint(argv):
    suite1 = unittest.TestLoader().loadTestsFromTestCase(testPackages)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(testModules)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(testFlows)
    suite_combo =  unittest.TestSuite([suite1, suite2, suite3])
    unittest.TextTestRunner(verbosity=2).run(suite_combo)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
