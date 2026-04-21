# CS 440 Spring 2026: Face and Digit Classification

This is the student template for the final project. Fill in each
`# TODO:` block and submit your code with a written report, following
the instructions below.

## 1. What you need to implement

Part 1(a): Perceptron (from scratch)
  * `q1a_perceptron_digits.py`
  * `q1a_perceptron_faces.py`

Part 1(b): 3 layer Neural Network (from scratch, numpy only)
  * `q1b_neural_net_scratch_digits.py`
  * `q1b_neural_net_scratch_faces.py`

Part 1(c): 3 layer Neural Network (with PyTorch)
  * `q1c_neural_net_pytorch_digits.py`
  * `q1c_neural_net_pytorch_faces.py`

Your network architecture must be **input, hidden1, hidden2, output**
(two hidden layers).

You must implement the core training and prediction logic yourself for
Parts 1(a) and 1(b). You may use numpy (and similar numerical
libraries) for linear algebra. You may not use sklearn, pytorch,
tensorflow, keras, or jax for the training, gradient computation, or
prediction in those parts. Part 1(c) is the only part where you are
expected to use PyTorch modules.

## 2. Repository layout

```
template/
  README.md                                this file
  team_details.json                        names and NetIDs go here
  report.pdf                               your written report
  util_digits.py                           data loading helpers (digits)
  util_faces.py                            data loading helpers (faces)
  q1a_perceptron_digits.py                 Part 1(a) digits
  q1a_perceptron_faces.py                  Part 1(a) faces
  q1b_neural_net_scratch_digits.py         Part 1(b) digits
  q1b_neural_net_scratch_faces.py          Part 1(b) faces
  q1c_neural_net_pytorch_digits.py         Part 1(c) digits
  q1c_neural_net_pytorch_faces.py          Part 1(c) faces
  q2q3_run_all_stats.py                         learning curve driver for the report
  digitdata/                               place the digit data files here
  facedata/                                place the face  data files here
```

Data files are distributed separately. Download the archive from

> http://rl.cs.rutgers.edu/fall2019/data.zip

and place its contents into the `digitdata/` and `facedata/`
directories so the following files exist:

```
digitdata/trainingimages     digitdata/traininglabels
digitdata/validationimages   digitdata/validationlabels
digitdata/testimages         digitdata/testlabels

facedata/facedatatrain        facedata/facedatatrainlabels
facedata/facedatavalidation   facedata/facedatavalidationlabels
facedata/facedatatest         facedata/facedatatestlabels
```

## 3. Running a single experiment

```
python3 q1a_perceptron_digits.py 50           # train on 50% of training data
python3 q1b_neural_net_scratch_faces.py 100   # train on 100% of training data
python3 q1c_neural_net_pytorch_digits.py 30   # etc.
```

## 4. Generating learning curve data for the report

```
python3 q2q3_run_all_stats.py                 # every classifier x every fraction
python3 q2q3_run_all_stats.py -i 3 -o results.json
python3 q2q3_run_all_stats.py -w perceptron_digits pytorch_faces
```

For each training fraction `p in {10, 20, ..., 100}` and each
classifier, the driver:

1. Draws `num_iterations` random subsamples of size `p%` of the
   training set.
2. Trains a fresh classifier on each subsample.
3. Evaluates it on the full test set.
4. Reports mean training time, mean error, and std of error.

These are exactly the numbers the project handout asks for in Part 3.

## 5. Report (Part 4)

Write a concise PDF report (recommended 4 to 8 pages) that covers:

1. Team and identification. Names and NetIDs of every team member.
2. Algorithm descriptions. For each of the three classifiers:
   * Model and architecture.
   * Loss function and update rule.
   * For the scratch NN: a derivation sketch of back propagation for
     your chosen activations.
   * Features used (pixel level is fine).
3. Experimental protocol. How you sampled training fractions, how
   many iterations, hyperparameters (learning rate, hidden sizes,
   epochs, batch size, activation).
4. Results. Two learning curve figures per dataset:
   * mean test error (with error bars of 1 std) vs training fraction.
   * mean training time vs training fraction.
5. Discussion. Which algorithm won on each dataset, how the three
   compare on accuracy and time, where the biggest variance came from,
   and the main lessons learned.
6. Optional: external resources or discussion partners consulted.

Save the report as `report.pdf` at the top level of your submission
folder. The report is worth 10 pts.

## 6. Submission instructions

### Group submissions

Only use the **name and NetID of the student submitting on behalf of
the group** for the top level directory. The names and NetIDs of
**all group members** must be listed in `team_details.json` and also
in the `report.pdf` file.

The submission folder should be named like `lastname-netid`, for
example `boyalakuntla-kb1204`. Zip that folder and upload the zip to
Canvas.

### File descriptions

* `team_details.json` contains identifying information for all
  members of the group:

  ```json
  {
    "names": ["student1-name", "student2-name", "..."],
    "netids": ["student1-netid", "student2-netid", "..."]
  }
  ```

* `report.pdf` is the written report described in Section 5.
* The six classifier files plus `util_digits.py`, `util_faces.py`, and
  `q2q3_run_all_stats.py`, all at the top level of the submission folder.

### Submission rules

1. One submission per team. The designated submitter uploads on
   Canvas.
2. Deadline: May 1st, 2026, 11:55 pm. No late submissions.
3. Before zipping, run a quick sanity check from a fresh shell:
   ```
   python3 q1a_perceptron_digits.py 10
   python3 q1b_neural_net_scratch_faces.py 10
   python3 q1c_neural_net_pytorch_digits.py 10
   ```
   All three should print a block of results without crashing. If any
   file crashes, you lose the runnability points for that module.
4. Do not include the `digitdata/` or `facedata/` directories in the
   zip; we will plug the data back in while grading.

## 7. Integrity

* Code and writing must be your team's own work. You may discuss
  ideas with other teams, but not share code or writing.
* Cite any external resources (tutorials, blog posts, textbooks) that
  you drew on.
* Do not peek at the test data during training.
* Plagiarism or unauthorised collaboration is not allowed.

Have fun!
