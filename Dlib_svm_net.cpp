// Dlib_svm_net.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;


int main()
{

	typedef matrix<double, 2, 1> sample_type;
	typedef radial_basis_kernel<sample_type> kernel_type;


	std::vector<sample_type> samples;
	std::vector<double> labels;

	for (int r = -20; r <= 20; ++r)
	{
		for (int c = -20; c <= 20; ++c)
		{
			sample_type samp;
			samp(0) = r;
			samp(1) = c;
			samples.push_back(samp);

			if (sqrt((double)r*r + c*c) <= 10)
				labels.push_back(+1);
			else
				labels.push_back(-1);

		}
	}


	vector_normalizer<sample_type> normalizer;
	normalizer.train(samples);
	for (unsigned long i = 0; i < samples.size(); ++i)
		samples[i] = normalizer(samples[i]);
	randomize_samples(samples, labels);

	svm_nu_trainer<kernel_type> trainer;
	trainer.set_kernel(kernel_type(0.15625));
	trainer.set_nu(0.15625);



	sample_type sample;
	typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
	typedef normalized_function<probabilistic_funct_type> pfunct_type;


	pfunct_type learned_pfunct;
	// Now let's open that file back up and load the function object it contains.
	deserialize("saved_function.dat") >> learned_pfunct;

	learned_pfunct.normalizer = normalizer;
	learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);


	typedef decision_function<kernel_type> dec_funct_type;
	typedef normalized_function<dec_funct_type> funct_type;
	funct_type learned_function;
	learned_function.normalizer = normalizer;  // save normalization information
	learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results


	cout << "\nnumber of support vectors in our learned_pfunct is "
		<< learned_pfunct.function.decision_funct.basis_vectors.size() << endl;

	do
	{
		cin>>sample(0);
		cin>>sample(1);
		cout << "the probability of SQRT(sample(0)2 + sample(1)2) to be <= 10 is " << learned_pfunct(sample) << endl;
		cout << "SQRT(sample(0)2 + sample(1)2) = " << sqrt((double)sample(0)*sample(0) + sample(1)*sample(1)) << endl;
		cout << "This is a +1 class example, the classifier output is " << learned_function(sample) << endl;
	} while (true);



}

