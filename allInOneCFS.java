import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.ExhaustiveSearch;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.pmml.Array;
import weka.core.pmml.jaxbbindings.True;
import weka.filters.Filter;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class allInOneCFS {

	private static final float PRUNE_CONFIDENCE = 0.25f;
	private static final int MIN_NUM_OBJ = 3;
	private static final boolean UNPRUNED = false;
	private static final int SEED = 7;
	private static String[] uselessFeatures = {"idauniq","indobyr"};
	private static int[] groupSizes = {1, 1, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
	private static int[] startIndex = {0, 1, 2, 5, 8, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 49, 53, 57, 61, 65, 72, 79, 86, 93, 100, 107, 114, 121, 128, 135, 142, 149, 156, 163, 170, 177, 184, 191, 198, 205, 212};
	private static int[] inGroup = {0, 1, 2, 2, 2, 5, 5, 5, 8, 9, 9, 9, 12, 12, 12, 15, 15, 15, 18, 18, 18, 21, 21, 21, 24, 24, 24, 27, 27, 27, 30, 30, 30, 33, 33, 33, 36, 36, 36, 39, 39, 39, 42, 42, 42, 45, 45, 45, 45, 49, 49, 49, 49, 53, 53, 53, 53, 57, 57, 57, 57, 61, 61, 61, 61, 65, 65, 65, 65, 65, 65, 65, 72, 72, 72, 72, 72, 72, 72, 79, 79, 79, 79, 79, 79, 79, 86, 86, 86, 86, 86, 86, 86, 93, 93, 93, 93, 93, 93, 93, 100, 100, 100, 100, 100, 100, 100, 107, 107, 107, 107, 107, 107, 107, 114, 114, 114, 114, 114, 114, 114, 121, 121, 121, 121, 121, 121, 121, 128, 128, 128, 128, 128, 128, 128, 135, 135, 135, 135, 135, 135, 135, 142, 142, 142, 142, 142, 142, 142, 149, 149, 149, 149, 149, 149, 149, 156, 156, 156, 156, 156, 156, 156, 163, 163, 163, 163, 163, 163, 163, 170, 170, 170, 170, 170, 170, 170, 177, 177, 177, 177, 177, 177, 177, 184, 184, 184, 184, 184, 184, 184, 191, 191, 191, 191, 191, 191, 191, 198, 198, 198, 198, 198, 198, 198, 205, 205, 205, 205, 205, 205, 205, 212, 212, 212, 212, 212, 212, 212, 219};


	private static int[][] twoPhaseResult = new int[][]{
		{0, 1, 3, 16, 26, 41, 43, 50, 73, 74, 88, 100, 101, 102, 123, 154, 158, 186, 188, 214, 219},
		{0, 1, 3, 25, 26, 41, 49, 50, 73, 74, 80, 88, 100, 101, 102, 121, 123, 158, 186, 206, 207, 214, 219},
		{0, 1, 3, 26, 41, 43, 49, 50, 73, 74, 80, 88, 100, 101, 102, 121, 123, 140, 154, 186, 188, 206, 214, 219},
		{0, 1, 3, 25, 26, 41, 43, 50, 73, 74, 80, 88, 100, 101, 102, 109, 121, 123, 141, 154, 186, 206, 214, 219},
		{0, 1, 3, 4, 8, 25, 32, 41, 42, 43, 49, 50, 70, 73, 74, 80, 88, 100, 101, 102, 121, 141, 154, 185, 186, 206, 214, 219},
		{0, 1, 3, 25, 26, 41, 42, 43, 50, 70, 73, 74, 80, 88, 101, 102, 107, 121, 123, 141, 158, 185, 186, 200, 214, 219},
		{0, 1, 3, 26, 41, 43, 49, 50, 73, 74, 77, 80, 88, 100, 101, 102, 121, 123, 139, 154, 158, 186, 206, 207, 219},
		{0, 1, 3, 11, 25, 32, 41, 49, 73, 74, 80, 88, 100, 101, 102, 123, 140, 154, 185, 186, 206, 207, 219},
		{0, 1, 3, 11, 26, 41, 43, 50, 72, 73, 74, 80, 100, 102, 121, 123, 140, 154, 158, 186, 206, 214, 219},
		{0, 1, 3, 41, 43, 70, 74, 88, 100, 101, 102, 121, 123, 154, 186, 214, 219}
	};
	private static int[][] cfsResult;
	private static Evaluation overallEval;

	private static String[] diseases = {"HeartAtt","Angina","Stroke","Diabetes","HBP","Dementia","Cataract","Arthritis","Osteoporosis","Parkinsons"};
	private static String[] allMethods = {"No_Feature_Selection", "standard_CFS", "Exh-CFS-Gr", "Exh-CFS-Gr+CFS", "WR-CFS"};
	private static String dataset = "elsa_w1-6_predict-w7HBP_7.5k";
	//	Real experiment
	private static String FOLD_NAME = "datasets/7496_folds/";
	private static String FILE_NAME = "datasets/" + dataset + ".arff";
	private static String disease,distribution,cfsMethod;
	private static int[] countSelectedFeatures = new int[220];

	// To test the code
//	private static String FILE_NAME = "datasets/elsa.arff";
//	private static String FOLD_NAME = "datasets/test_folds/";

	private static final double[] WITHIN_GROUP_WEIGHTS = {0.5, 0.6, 0.7, 0.8, 0.9};
	private static double wWG, wAG;
	private static String[] originalTypes  = {"indsex", "w6indager", "w2apoe", "w2", "w4", "w6"};
	private static String[] constructedTypes = {"diff_w24", "diff_w46", "diff_w26", "mono_w246",  "up_w24",  "up_w46"};
	private static StringBuilder sbFeatureCount = new StringBuilder("Disease,Class Distribution,CFS Method,");
	private static StringBuilder sbCountAll = new StringBuilder("Disease,Class Distribution,CFS Method,");;
	private static StringBuilder sbPredictiveAccuracyJ48 = new StringBuilder("Disease,Class Distribution,CFS Method,Precision,Recall,F-Measure,ROC Area\n");
	private static StringBuilder sbFMeasureJ48 = new StringBuilder("Disease,Class Distribution," + allMethods[0] + ',' + allMethods[1] + ',' + allMethods[2] + ',' + allMethods[3] + ',' + allMethods[4]);
	private static StringBuilder sbPredictiveAccuracyNB = new StringBuilder(sbPredictiveAccuracyJ48);
	private static StringBuilder sbFMeasureNB = new StringBuilder(sbFMeasureJ48);

	private static final PrintStream CONSOLE = System.out;
	private static boolean isOriginalCfs;
	private static final int START_BIAS = 8;
	private static final boolean SHORT_RUN = false;
	private static final int NUM_FOLDS = 10;
	private static String classificationAlgo = "WR-CFS_only";
	private static String PATH_OUT_DEFAULT = "results/" + classificationAlgo + "/";

	private static ArrayList<int[]> countWeightCandidates = new ArrayList<>();
	private static boolean isJ48;
	private static Instances[] trainSet = new Instances[NUM_FOLDS];
	private static Instances[] testSet = new Instances[NUM_FOLDS];
	private static Instances[] samplingTrainSet = new Instances[NUM_FOLDS];
	private static int[] globalPhase1;


	private static void setUpSb() {
		for (String str : originalTypes) { sbFeatureCount.append(str); sbFeatureCount.append(','); }
		for (String str : constructedTypes) { sbFeatureCount.append(str); sbFeatureCount.append(','); }
		sbFeatureCount.append("Original Total,Constructed Total,All Total\n");
		sbCountAll.append("indsex,w6indager,w2clotb,w4clotb,w6clotb,w2fit,w4fit,w6fit,w2apoe,w2hasurg,w4hasurg,w6hasurg,w2eyesurg,w4eyesurg,w6eyesurg,w2hastro,w4hastro,w6hastro,w2chestin,w4chestinf,w6chestinf,w2inhaler,w4inhaler,w6inhaler,w2mmssre,w4mmssre,w6mmssre,w2mmstre,w4mmstre,w6mmstre,w2mmftre2,w4mmftre2,w6mmftre2,w2mmlore,w4mmlore,w6mmlore,w2mmlsre,w4mmlsre,w6mmlsre,w2mmcrre,w4mmcrre,w6mmcrre,w2mmrroc,w4mmrroc,w6mmrroc,w2hipval,w4hipval,hipval_up_w24,hipval_diff_w24,w2whval,w4whval,whval_up_w24,whval_diff_w24,w2htpf,w4htpf,htpf_up_w24,htpf_diff_w24,w4wbc,w6wbc,wbc_up_w46,wbc_diff_w46,w4mch,w6mch,mch_up_w46,mch_diff_w46,w2sysval,w4sysval,w6sysval,sysval_mono_w246,sysval_diff_w24,sysval_diff_w46,sysval_diff_w26,w2diaval,w4diaval,w6diaval,diaval_mono_w246,diaval_diff_w24,diaval_diff_w46,diaval_diff_w26,w2pulval,w4pulval,w6pulval,pulval_mono_w246,pulval_diff_w24,pulval_diff_w46,pulval_diff_w26,w2mapval,w4mapval,w6mapval,mapval_mono_w246,mapval_diff_w24,mapval_diff_w46,mapval_diff_w26,w2cfib,w4cfib,w6cfib,cfib_mono_w246,cfib_diff_w24,cfib_diff_w46,cfib_diff_w26,w2chol,w4chol,w6chol,chol_mono_w246,chol_diff_w24,chol_diff_w46,chol_diff_w26,w2hdl,w4hdl,w6hdl,hdl_mono_w246,hdl_diff_w24,hdl_diff_w46,hdl_diff_w26,w2trig,w4trig,w6trig,trig_mono_w246,trig_diff_w24,trig_diff_w46,trig_diff_w26,w2ldl,w4ldl,w6ldl,ldl_mono_w246,ldl_diff_w24,ldl_diff_w46,ldl_diff_w26,w2fglu,w4fglu,w6fglu,fglu_mono_w246,fglu_diff_w24,fglu_diff_w46,fglu_diff_w26,w2rtin,w4rtin,w6rtin,rtin_mono_w246,rtin_diff_w24,rtin_diff_w46,rtin_diff_w26,w2hscrp,w4hscrp,w6hscrp,hscrp_mono_w246,hscrp_diff_w24,hscrp_diff_w46,hscrp_diff_w26,w2hgb,w4hgb,w6hgb,hgb_mono_w246,hgb_diff_w24,hgb_diff_w46,hgb_diff_w26,w2hba1c,w4hba1c,w6hba1c,hba1c_mono_w246,hba1c_diff_w24,hba1c_diff_w46,hba1c_diff_w26,w2htval,w4htval,w6htval,htval_mono_w246,htval_diff_w24,htval_diff_w46,htval_diff_w26,w2wtval,w4wtval,w6wtval,wtval_mono_w246,wtval_diff_w24,wtval_diff_w46,wtval_diff_w26,w2bmival,w4bmival,w6bmival,bmival_mono_w246,bmival_diff_w24,bmival_diff_w46,bmival_diff_w26,w2wstval,w4wstval,w6wstval,wstval_mono_w246,wstval_diff_w24,wstval_diff_w46,wstval_diff_w26,w2htfvc,w4htfvc,w6htfvc,htfvc_mono_w246,htfvc_diff_w24,htfvc_diff_w46,htfvc_diff_w26,w2htfev,w4htfev,w6htfev,htfev_mono_w246,htfev_diff_w24,htfev_diff_w46,htfev_diff_w26,w2mmgsd_me,w4mmgsd_me,w6mmgsd_me,mmgsd_me_mono_w246,mmgsd_me_diff_w24,mmgsd_me_diff_w46,mmgsd_me_diff_w26,w2mmgsn_me,w4mmgsn_me,w6mmgsn_me,mmgsn_me_mono_w246,mmgsn_me_diff_w24,mmgsn_me_diff_w46,mmgsn_me_diff_w26\n");
	}

	private static void writeAll(String pathOut) throws Exception {
		new File(pathOut).mkdirs();
		PrintWriter pw = new PrintWriter(new File(pathOut+"count_feature_types.csv"));
		pw.write(sbFeatureCount.toString());
		pw.close();

		pw = new PrintWriter(new File(pathOut+"selected_feature_count.csv"));
		pw.write(sbCountAll.toString());
		pw.close();

		pw = new PrintWriter(new File(pathOut+"predictive_accuracy_J48.csv"));
		pw.write(sbPredictiveAccuracyJ48.toString());
		pw.close();

		pw = new PrintWriter(new File(pathOut+"f-measures_J48.csv"));
		pw.write(sbFMeasureJ48.toString());
		pw.close();

		pw = new PrintWriter(new File(pathOut+"predictive_accuracy_NB.csv"));
		pw.write(sbPredictiveAccuracyNB.toString());
		pw.close();

		pw = new PrintWriter(new File(pathOut+"f-measures_NB.csv"));
		pw.write(sbFMeasureNB.toString());
		pw.close();

//		System.setOut(new PrintStream(new FileOutputStream(pathOut+"weight_candidates.csv")));
//		for(double w : WITHIN_GROUP_WEIGHTS)	System.out.print(w + ",");
//		for(int[] counts : countWeightCandidates) {
//			System.out.println();
//			for(int c : counts)	System.out.print(c + ",");
//		}
//		System.setOut(CONSOLE);
	}

	private static Instances loadDataSet(String filename) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(filename));
		return loader.getDataSet();
	}
	private static Instances doFilter(Instances instances,
			ArrayList<Integer> toKeep) {

		Instances ret = new Instances(instances);

		for (int i = instances.numInstances() - 1; i >= 0; i--) {
			if(!toKeep.contains(i)){
				ret.delete(i);
			}
		}

		return ret;
	}


	private static Instances buildDS(Instances instances, int foldId, boolean isTrain) throws FileNotFoundException {
		String which = isTrain ? "train" : "test";
		String foldFileName = FOLD_NAME + foldId + "." + which + ".fold";

		ArrayList<Integer> toKeep = new ArrayList<Integer>();
		Scanner in = new Scanner(new FileReader(foldFileName));
		in.useDelimiter(";");
		while(in.hasNextInt()){
			toKeep.add(in.nextInt());
		}
		in.close();

		Instances crossInstances = doFilter(instances, toKeep);
		return crossInstances;
	}

	private static Instances removeDifficultFeatures(Instances instances)
			throws Exception {
		String[] toRemove = uselessFeatures;

		Remove remove = new Remove();
		int[] attributes = new int[toRemove.length];

		for (int i = 0; i < toRemove.length; i++) {
			attributes[i] = instances.attribute(toRemove[i]).index();
		}

		remove.setAttributeIndicesArray(attributes);
		remove.setInputFormat(instances);
		Instances ret = Filter.useFilter(instances, remove);
		return ret;
	}

	private static Instances getSelectedAttributes(Instances data, int[] indices) throws Exception {
		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indices);
		remove.setInvertSelection(true);
		remove.setInputFormat(data);
		return Filter.useFilter(data, remove);
	}

	public static void lineBreak(char c, int n) {
		for(int i=0 ; i<n ; i++) System.out.print(c);
		System.out.println();
	}

	public static void toFile(String pathOut, String text) throws Exception {

		PrintWriter writer = new PrintWriter(pathOut);
		writer.println(text);
		writer.close();
	}

	public static String toFeatureString(Instances data, int[] indices) {
		int len = indices.length;
		if(indices[len-1]==data.classIndex())	len--;
		String[] features = new String[len];
		for(int i=0; i<len ; i++) {
			int index = indices[i];
			features[i] = data.attribute(index).name();
		}
		return Arrays.toString(features);
	}

	// a successive set of features
	public static String toFeatureString(Instances data, int first, int len) {
		String[] features = new String[len];
		for(int i=0, index=first; i<len ; i++) {
			features[i] = data.attribute(index++).name();
		}
		return Arrays.toString(features);
	}

	public static void displayCfsSummary(Instances data, int[][] result) {
		int[] count = new int[data.numAttributes()];
		boolean[] isChosen = new boolean[groupSizes.length];

		for (int n = 0; n < NUM_FOLDS; n++) {
			int[] indices = result[n];
			for (int i = 0; i < indices.length; i++) {
				count[indices[i]]++;
			}
		}

		for(int i=0, index=0 ; i<groupSizes.length ; i++) {
			int len = groupSizes[i];
			for(int j=0 ; j<len ; j++, index++) {
				if(count[index]>0) {
					isChosen[i] = true;
				}
			}
		}

		String[] conceptualFeatures = {"indsex", "indager", "clotb", "fit", "apoe", "hasurg", "eyesurg", "hastro", "chestin", "inhaler", "mmssre", "mmstre", "mmftre2", "mmlore", "mmlsre", "mmcrre", "mmrroc","hipval","whval","htpf","wbc","mch","sysval","diaval","pulval","mapval","cfib","chol","hdl","trig","ldl","fglu","rtin","hscrp","hgb","hba1c","htval","wtval","bmival","wstval","htfvc","htfev","mmgsd_me","mmgsn_me"};

		displayShortCfs(data,result);
		System.out.println("\nSummary->");
		for(int i=0, index=0 ; i<groupSizes.length ; i++) {
			int len = groupSizes[i];
//			if(phase1[i].length>0)	System.out.println("phase1: (" + len + ")" + toFeatureString(data,index,len));
			if(isChosen[i]) {
				System.out.println(conceptualFeatures[i] + ":(" + len + ")" + toFeatureString(data,index,len));
				System.out.print("Selected:[ ");
				for(int j=0, k=index ; j<len ; j++, k++) {
					if(count[k]>0)	System.out.print(data.attribute(k).name() + "(" + count[k] + "/10) ");
				}
				System.out.println("]\n");
			}
			index+=len;
		}
	}

	public static void displayVeryShortCfs(int[][] result) {
		for(int i=0 ; i<result.length ; i++) {
			System.out.println(result[i].length);
			for(int j=0 ; j<result[i].length ; j++)	System.out.print(result[i][j] + ",");
			System.out.println();
		}
	}

	public static void displayShortCfs(Instances data, int[][] result) {
		System.out.println("\n*** " + cfsMethod + " result ***");
		for (int n = 0; n < NUM_FOLDS; n++) {
			int[] indices = result[n];
			System.out.println("fold(" + n + "): {" + String.format("%2d",indices.length-1) +  " features}->" + toFeatureString(data, indices));
		}
	}

	public static void displayLongCfs(Instances data, int[][] result) {
		int avgResult = 0;
		for (int n = 0; n < NUM_FOLDS; n++) {
			int[] indices = result[n];
			avgResult += indices.length;
			System.out.println("\nfold(" + n + "):");
			System.out.println("\nSelected( " + (indices.length-1) + " ): " + toFeatureString(data, indices));
			lineBreak('=', 80);
		}
		System.out.println("The average no. selected features = " + (avgResult/(float) NUM_FOLDS-1));
	}

	private static Evaluation runClassification(Classifier classifier, Instances train, Instances test) throws Exception {
		Evaluation eval = new Evaluation(test);
		classifier.buildClassifier(train);
		eval.evaluateModel(classifier, test);
		overallEval.evaluateModel(classifier, test);

		return eval;
	}

	public static void displayPredictiveAccuracyMeasures(Evaluation[] evals, String[] models, String pathOut) throws Exception {

		for(int i=0 ; i<NUM_FOLDS ; i++) {
			Evaluation eval = evals[i];
			System.out.println("FOLD " + i + ":");
			System.out.print(eval.toHappyClassDetailsString());
			lineBreak('=',64);
			toFile(pathOut+i+".txt", models[i] + toEvaluationString(eval));
		}
		System.out.print(overallEval.toHappyClassDetailsString("*** Overall Result *** "));
		lineBreak('#',64);

		if(isJ48) {
			sbPredictiveAccuracyJ48.append(overallEval.toHappyOverallString(disease+','+distribution+','+cfsMethod+','));
			sbFMeasureJ48.append(String.format(",%.3f", overallEval.unWeightedFMeasure()));
		}
		else {
			sbPredictiveAccuracyNB.append(overallEval.toHappyOverallString(disease+','+distribution+','+cfsMethod+','));
			sbFMeasureNB.append(String.format(",%.3f", overallEval.unWeightedFMeasure()));
		}
	}

	private static String toEvaluationString(Evaluation eval) throws Exception {
		return eval.toMatrixString() + eval.toHappyClassDetailsString();
	}

	private static Instances undersampling(Instances data, double bias) throws Exception {
		SpreadSubsample filter = new SpreadSubsample();
		filter.setDistributionSpread(bias);
		filter.setInputFormat(data);
		return Filter.useFilter(data, filter);
	}


	private static void showClassDistribution(Instances data) {
		int[] a = new int[2];
		for(int c = 0; c < data.attributeStats(data.numAttributes() - 1).nominalCounts.length; c++)
		{
			int cn = data.attributeStats(data.numAttributes() - 1).nominalCounts[c];
			System.out.printf("Class distribution for %s: %d \n" ,data.classAttribute().value(c) ,cn);
			a[c] = cn;
//			System.out.printf("A-priori class distribution for %s: %.4f \n", data.classAttribute().value(c), ((double)cn) / data.numInstances());

		}
		distribution = a[0] + " to " + a[1];
	}

	protected static int[] runFeatureSelection(Instances train, ASSearch search) throws Exception {
		AttributeSelection attsel = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		if(!isOriginalCfs) {
			eval.setInGroup(inGroup, wWG, wAG);
		}
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		attsel.SelectAttributes(train);
		int[] indices = attsel.selectedAttributes();
		return indices;
	}

	public static int[] twoPhaseCfs(Instances train, ASSearch search) throws Exception {
		IntStream.Builder groupBuilder = IntStream.builder();
		for(int i=0, index=0 ; i<groupSizes.length ; i++){
			int len = groupSizes[i];
			int[] indices = IntStream.rangeClosed(index, index+len).toArray();
			indices[len] = train.classIndex();
			Instances subTrain = getSelectedAttributes(train,indices);
			int[] group = runFeatureSelection(subTrain, new ExhaustiveSearch());
			for(int j=0 ; j<group.length-1 ; j++)	groupBuilder.accept(group[j]+index);
			index += len;
		}
		groupBuilder.accept(train.classIndex());
		int[] innerphaseIndices = groupBuilder.build().toArray();
//			System.out.println(Arrays.toString(innerphaseIndices));
		Instances reducedData = getSelectedAttributes(train,innerphaseIndices);
		int[] outterphaseIndices = runFeatureSelection(reducedData, search);
		// correct the indices: inner index -> outter index -> real dataset
		for(int i=0 ; i<outterphaseIndices.length ; i++) outterphaseIndices[i] = innerphaseIndices[outterphaseIndices[i]];
//			System.out.println(Arrays.toString(outterphaseIndices));
		globalPhase1 = innerphaseIndices;
		return outterphaseIndices;
	}

	private static void classification(Classifier classifier, Instances data, int[][] result, String pathOut) throws Exception {
		Evaluation[] evals = new Evaluation[NUM_FOLDS];
		String[] trees = new String[NUM_FOLDS];
		overallEval = new Evaluation(data);
		isJ48 = classifier instanceof J48;
		classificationAlgo = isJ48 ? "J48" : "NaiveBayes";
		pathOut += classificationAlgo + '/';
		new File(pathOut).mkdirs();
		long startTime = System.nanoTime();

		System.out.print("\nRunning " + classificationAlgo + " to " + pathOut + " ==> ");
		for (int i = 0; i < NUM_FOLDS; i++) {
			System.out.print(i + " . . ");
			Instances train = samplingTrainSet[i];
			Instances test = testSet[i];
			if(result!=null) {
				train = getSelectedAttributes(train, result[i]);
				test  = getSelectedAttributes(test, result[i]);
			}
			Evaluation eval = runClassification(classifier, train, test);
			evals[i] = eval;
			trees[i] = classifier.toString();
		}
		System.out.println("FINISHED!");

		long stopTime = System.nanoTime();
		long seconds = TimeUnit.SECONDS.convert((stopTime-startTime), TimeUnit.NANOSECONDS);

		System.setOut(new PrintStream(new FileOutputStream(pathOut+"overall.txt")));
		displayPredictiveAccuracyMeasures(evals,trees,pathOut);
		System.setOut(CONSOLE);
		System.out.println("Time taken: " + seconds + " seconds");

		// build the final model
		classifier.buildClassifier(data);
		String finalModel = classifier.toString();
		toFile(pathOut+"finalModel.txt", finalModel);
	}

	public static int[] cfsChoices(Instances train, boolean isTwoPhase, ASSearch search) throws Exception {
		int[] cfsResult, indices;
		if (isTwoPhase) {
			cfsResult = twoPhaseCfs(train, search);
		}
		else if(isOriginalCfs) {
			cfsResult = runFeatureSelection(train, search);
		}
		else {
			int[] count = new int[train.numAttributes()];
			ArrayList<Integer> major = new ArrayList<>();
			for (int n = 0; n < WITHIN_GROUP_WEIGHTS.length; n++) {
				wWG = WITHIN_GROUP_WEIGHTS[n];
				wAG = 1.0 - wWG;
				GreedyStepwise subSearch = new GreedyStepwise();
				indices = runFeatureSelection(train, subSearch);
				for(int index: indices)	count[index]++;
			}
			System.out.println(Arrays.toString(count));
			for(int i=0 ; i<count.length ; i++)
				if(count[i]>=3)
					major.add(i);
			cfsResult = major.stream().mapToInt(i -> i).toArray();
			System.out.println(Arrays.toString(cfsResult));
		}
		return cfsResult;
	}

	public static void crossValidation(boolean isTwoPhase, Instances data, String pathOut) throws Exception  {
		long startTime = System.nanoTime();
		int[][] result = new int[NUM_FOLDS][];
		int[][] phase1 = new int[NUM_FOLDS][];

		System.out.print("\nRunning " + cfsMethod + " to " + pathOut + " ==> ");

		for (int i = 0; i < NUM_FOLDS; i++) {
			System.out.print(i + " . . ");
			Instances train = samplingTrainSet[i];
			ASSearch search = new GreedyStepwise();
			result[i] = cfsChoices(train, isTwoPhase, search);
			if(isTwoPhase) phase1[i] = globalPhase1;
		}
		System.out.println("DONE!");
		Long stopTime = System.nanoTime();
		long seconds = TimeUnit.SECONDS.convert((stopTime-startTime), TimeUnit.NANOSECONDS);

		int[] resultAll = cfsChoices(data, isTwoPhase, new GreedyStepwise());
		Instances fsData = getSelectedAttributes(data, resultAll);

		if(isTwoPhase)  {
			Instances phase1FSData = getSelectedAttributes(data, globalPhase1);
			cfsMethod = allMethods[2];
			String localPath = pathOut + cfsMethod + '/';
			new File(localPath).mkdirs();
			System.setOut(new PrintStream(new FileOutputStream(localPath+"cfs.txt")));
			displayLongCfs(data, phase1);
			displayCfsSummary(data, phase1);
			System.out.println("Time taken: " + seconds + " seconds");

			System.setOut(new PrintStream(new FileOutputStream(localPath+"indices_cfs.txt")));
			displayVeryShortCfs(phase1);
			System.setOut(CONSOLE);
			countSelectedFeatures(data, phase1);

			J48 classifier = new J48();
			classifier.happySetOptions(MIN_NUM_OBJ, UNPRUNED, PRUNE_CONFIDENCE);
			classification(classifier, phase1FSData, phase1, localPath);
			classification(new NaiveBayes(), phase1FSData, phase1, localPath);
			// setting for next step
			cfsMethod = allMethods[3];
			pathOut += cfsMethod + '/';
		}

		new File(pathOut).mkdirs();
		System.setOut(new PrintStream(new FileOutputStream(pathOut+"cfs.txt")));
		displayLongCfs(data, result);
		displayCfsSummary(data, result);
		System.out.println("Time taken: " + seconds + " seconds");

		System.setOut(new PrintStream(new FileOutputStream(pathOut+"indices_cfs.txt")));
		displayVeryShortCfs(result);

		System.setOut(CONSOLE);
		System.out.println("Time taken: " + seconds + " seconds");

		// TODO: Count the number of selected original and constructed features
		countSelectedFeatures(data, result);

		J48 classifier = new J48();
		classifier.happySetOptions(MIN_NUM_OBJ, UNPRUNED, PRUNE_CONFIDENCE);

		classification(classifier, fsData, result, pathOut);
		classification(new NaiveBayes(), fsData, result, pathOut);
	}

	private static int featureType(String feature, String[] types) {
		for(int k=0 ; k<types.length ; k++)
			if(feature.contains(types[k]))
				return k;
//		System.out.println(feature);
		return -1;
	}

	private static void countSelectedFeatures(Instances data, int[][] result) throws Exception {
		int[] countOriginal = new int[originalTypes.length];
		int[] countConstruct = new int[constructedTypes.length];
		int[] countAll = new int[data.numAttributes()-1];

		for(int i=0 ; i<NUM_FOLDS ; i++) {
			for(int j=0 ; j<result[i].length-1 ; j++) {
				countAll[result[i][j]]++;
				String feature = data.attribute(result[i][j]).name();
				int index = featureType(feature, constructedTypes);
				if(index>0) 	countConstruct[index]++;
				else 			countOriginal[featureType(feature, originalTypes)]++;
			}
		}
		sbFeatureCount.append(disease+','+distribution+','+cfsMethod+',');
		sbCountAll.append(disease+','+distribution+','+cfsMethod);
		int sumOriginal=0, sumConstruct=0;
		for (int count : countOriginal) {
			sumOriginal += count;
			sbFeatureCount.append(count); sbFeatureCount.append(',');
		}
		for (int count : countConstruct) {
			sumConstruct += count;
			sbFeatureCount.append(count); sbFeatureCount.append(',');
		}
		sbFeatureCount.append(sumOriginal); sbFeatureCount.append(',');
		sbFeatureCount.append(sumConstruct); sbFeatureCount.append(',');
		sbFeatureCount.append(sumOriginal+sumConstruct); sbFeatureCount.append('\n');

		for(int count : countAll) {
			sbCountAll.append(','); sbCountAll.append(count);
		}
		sbCountAll.append('\n');
	}


	/**
	 * Main function for the feature selection
	 * @throws Exception
	 */
	public static void featureSelection(String pathOut) throws Exception {
		// load dataset
		Instances instances = loadDataSet(FILE_NAME);
		instances.setClassIndex(instances.numAttributes() - 1);
		Instances instancesAll = new Instances(instances);
		instancesAll = removeDifficultFeatures(instancesAll);

		prepareCrossValidationDataFolds(instancesAll);
		Instances data = instancesAll;
		String folder = "default_distribution/";
		int bias=START_BIAS;
		do {
			if(instancesAll.numInstances()>data.numInstances() || bias==8) {
				new File(pathOut+folder).mkdirs();
				showClassDistribution(data);
				sbFMeasureJ48.append('\n' + disease + ',' + distribution);
				sbFMeasureNB.append('\n' + disease + ',' + distribution);

//				isOriginalCfs = true;
//				// No feature selection
//				J48 classifier = new J48();
//				classifier.happySetOptions(MIN_NUM_OBJ, UNPRUNED, PRUNE_CONFIDENCE);
//				cfsMethod = allMethods[0];
//				classification(classifier, data, null, pathOut+folder+cfsMethod+"/");
//				classification(new NaiveBayes(), data, null, pathOut+folder+cfsMethod+"/");
//
//				// run CFS
//				cfsMethod = allMethods[1];
//				if(!SHORT_RUN) crossValidation( false, data, pathOut+folder+cfsMethod+"/");
//				// Exh-CFS-Gr and Exh-CFS-Gr+CFS
//				cfsMethod = allMethods[3];
//				crossValidation( true, data, pathOut+folder);
				// weighted redundancy cfs
				isOriginalCfs = false;
				cfsMethod = allMethods[4];
				crossValidation( false, data, pathOut+folder+cfsMethod+"/");
			}
			bias/=2;
			folder = "undersampling_" + bias +  "-1/";
			data = prepareTrainSet(instancesAll, bias);
		} while(bias>0);

	}


	private static void prepareCrossValidationDataFolds(Instances data) throws Exception {
		Random rand = new Random(SEED);
		Instances randData = new Instances(data);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(NUM_FOLDS);

		// perform cross-validation
		System.out.println();
		System.out.println("=== Setup ===");
		System.out.println("Dataset: " + data.relationName());
		System.out.println("Folds: " + NUM_FOLDS);
		System.out.println("Seed: " + SEED);
		System.out.println();

		for (int i = 0; i < NUM_FOLDS; i++) {
			trainSet[i] = randData.trainCV(NUM_FOLDS, i);
			testSet[i] = randData.testCV(NUM_FOLDS, i);
		}
		samplingTrainSet = trainSet;

//		for (int i = 0; i < NUM_FOLDS; i++) {
//			trainSet[i] = buildDS(data, i, true);
//			testSet[i] = buildDS(data, i, false);
//		}
	}

	private static Instances prepareTrainSet(Instances instancesAll, int bias) throws Exception {
		for (int i = 0; i < NUM_FOLDS; i++) {
			samplingTrainSet[i] = undersampling(trainSet[i],bias);
		}
		return undersampling(instancesAll, bias);
	}

	public static void test() throws Exception {
		// load dataset
		Instances instances = loadDataSet(FILE_NAME);
		instances.setClassIndex(instances.numAttributes() - 1);
		Instances instancesAll = new Instances(instances);
		instancesAll = removeDifficultFeatures(instancesAll);

		Instances samplingData = undersampling(instancesAll, 1);

		isOriginalCfs = true;
		GreedyStepwise searchA = new GreedyStepwise();
		int[] temp = cfsChoices(samplingData, false, searchA);


		isOriginalCfs = false;
		cfsChoices(samplingData, false, searchA);
		System.out.println(Arrays.toString(temp));
//		GreedyStepwise searchB = new GreedyStepwise();
//		cfsChoices(samplingData, true, searchB);
//		System.out.println("glob merit = " + globalMerit);

	}



	public static void main(String args[]) throws Exception {
		// set up string builders
//		setUpSb();
//		for(String target_disease : diseases) {
//			long startTime = System.nanoTime();
//
//			dataset = "elsa_w1-6_predict-w7" + target_disease + "_7.5k";
//			FOLD_NAME = "datasets/7496_folds/";
//			FILE_NAME = "datasets/" + dataset + ".arff";
//			disease = target_disease;
//			featureSelection(PATH_OUT_DEFAULT + dataset + '/');
//
//			long stopTime = System.nanoTime();
//			long seconds = TimeUnit.SECONDS.convert((stopTime-startTime), TimeUnit.NANOSECONDS);
//			System.out.println("*** it took: " + seconds + " seconds for processing " + target_disease);
//		}
//		writeAll(PATH_OUT_DEFAULT);
		test();
	}


}
