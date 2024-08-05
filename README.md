<div class="container" id="notebook-container">

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="What-is-Model-Validation" tabindex="0">What is Model Validation<a class="anchor-link" href="https://www.kaggle.com/code/dansbecker/model-validation#What-is-Model-Validation" target="_self" rel=" noreferrer nofollow">¶</a></h1><p>You'll want to evaluate almost every model you ever build. In most (though not all) applications, the relevant measure of model quality is predictive accuracy. In other words, will the model's predictions be close to what actually happens.</p>
<p>Many people make a huge mistake when measuring predictive accuracy. They make predictions with their <em>training data</em> and compare those predictions to the target values in the <em>training data</em>. You'll see the problem with this approach and how to solve it in a moment, but let's think about how we'd do this first.</p>
<p>You'd first need to summarize the model quality into an understandable way. If you compare predicted and actual home values for 10,000 houses, you'll likely find mix of good and bad predictions. Looking through a list of 10,000 predicted and actual values would be pointless. We need to summarize this into a single metric.</p>
<p>There are many metrics for summarizing model quality, but we'll start with one called <strong>Mean Absolute Error</strong> (also called <strong>MAE</strong>). Let's break down this metric starting with the last word, error.</p>
<p>The prediction error for each house is: <br></p>

<pre><code>error=actual−predicted</code></pre>
<p>So, if a house cost <span>$</span>150,000 and you predicted it would cost <span>$</span>100,000 the error is <span>$</span>50,000.</p>
<p>With the MAE metric, we take the absolute value of each error. This converts each error to a positive number. We then take the average of those absolute errors. This is our measure of model quality. In plain English, it can be said as</p>
<blockquote><p>On average, our predictions are off by about X.</p>
</blockquote>
<p>To calculate MAE, we first need a model. That is built in a hidden cell below, which you can review by clicking the <code>code</code> button.</p>

</div>
</div>
</div>
    </div>

    <div class="_kg_hide-input-true _kg_hide-output-true " role="section" tabindex="0"><div id="visibility-toggle-portal-1" class="visibility-toggle-portal"><div class="sc-exayXG bsNdS"><div class="sc-kyMESl jnHWTw" style="flex-grow: 1; margin-right: 16px;"></div><div aria-live="polite" role="button" tabindex="0" class="sc-dorvvM fOAckL"><i class="rmwc-icon rmwc-icon--ligature google-symbols sc-dJDBYC ddUugC notranslate sc-kTYLvb goFMXh google-symbols" aria-hidden="true">unfold_more</i>Show hidden cell</div><div class="sc-kyMESl jnHWTw" style="flex-grow: 1; margin-left: 16px;"></div></div></div>
        
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell"><div id="sharing-control-portal-2" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Data Loading Code Hidden Here</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>

<span class="c1"># Load data</span>
<span class="n">melbourne_file_path</span> <span class="o">=</span> <span class="s1">'../input/melbourne-housing-snapshot/melb_data.csv'</span>
<span class="n">melbourne_data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">melbourne_file_path</span><span class="p">)</span> 
<span class="c1"># Filter rows with missing price values</span>
<span class="n">filtered_melbourne_data</span> <span class="o">=</span> <span class="n">melbourne_data</span><span class="o">.</span><span class="n">dropna</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="c1"># Choose target and features</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">filtered_melbourne_data</span><span class="o">.</span><span class="n">Price</span>
<span class="n">melbourne_features</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'Rooms'</span><span class="p">,</span> <span class="s1">'Bathroom'</span><span class="p">,</span> <span class="s1">'Landsize'</span><span class="p">,</span> <span class="s1">'BuildingArea'</span><span class="p">,</span> 
                        <span class="s1">'YearBuilt'</span><span class="p">,</span> <span class="s1">'Lattitude'</span><span class="p">,</span> <span class="s1">'Longtitude'</span><span class="p">]</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">filtered_melbourne_data</span><span class="p">[</span><span class="n">melbourne_features</span><span class="p">]</span>

<span class="kn">from</span> <span class="nn">sklearn.tree</span> <span class="kn">import</span> <span class="n">DecisionTreeRegressor</span>
<span class="c1"># Define model</span>
<span class="n">melbourne_model</span> <span class="o">=</span> <span class="n">DecisionTreeRegressor</span><span class="p">()</span>
<span class="c1"># Fit model</span>
<span class="n">melbourne_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[1]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>DecisionTreeRegressor()</pre>
</div>

</div>

</div>
</div>

</div>
    </div>

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell"><div id="sharing-control-portal-3" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
<div class="text_cell_render border-box-sizing rendered_html">
<p>Once we have a model, here is how we calculate the mean absolute error:</p>

</div>
</div>
</div>
    </div>

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell"><div id="sharing-control-portal-4" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">mean_absolute_error</span>

<span class="n">predicted_home_prices</span> <span class="o">=</span> <span class="n">melbourne_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
<span class="n">mean_absolute_error</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="n">predicted_home_prices</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[2]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>434.71594577146544</pre>
</div>

</div>

</div>
</div>

</div>
    </div>

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell"><div id="sharing-control-portal-5" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="The-Problem-with-%22In-Sample%22-Scores" tabindex="0">The Problem with "In-Sample" Scores<a class="anchor-link" href="https://www.kaggle.com/code/dansbecker/model-validation#The-Problem-with-%22In-Sample%22-Scores" target="_self" rel=" noreferrer nofollow">¶</a></h1><p>The measure we just computed can be called an "in-sample" score. We used a single "sample" of houses for both building the model and evaluating it. Here's why this is bad.</p>
<p>Imagine that, in the large real estate market, door color is unrelated to home price.</p>
<p>However, in the sample of data you used to build the model, all homes with green doors were very expensive. The model's job is to find patterns that predict home prices, so it will see this pattern, and it will always predict high prices for homes with green doors.</p>
<p>Since this pattern was derived from the training data, the model will appear accurate in the training data.</p>
<p>But if this pattern doesn't hold when the model sees new data, the model would be very inaccurate when used in practice.</p>
<p>Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called <strong>validation data</strong>.</p>
<h1 id="Coding-It" tabindex="0">Coding It<a class="anchor-link" href="https://www.kaggle.com/code/dansbecker/model-validation#Coding-It" target="_self" rel=" noreferrer nofollow">¶</a></h1><p>The scikit-learn library has a function <code>train_test_split</code> to break up the data into two pieces. We'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate <code>mean_absolute_error</code>.</p>
<p>Here is the code:</p>

</div>
</div>
</div>
    </div>

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell"><div id="sharing-control-portal-6" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>

<span class="c1"># split data into training and validation data, for both features and target</span>
<span class="c1"># The split is based on a random number generator. Supplying a numeric value to</span>
<span class="c1"># the random_state argument guarantees we get the same split every time we</span>
<span class="c1"># run this script.</span>
<span class="n">train_X</span><span class="p">,</span> <span class="n">val_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">val_y</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">random_state</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>
<span class="c1"># Define model</span>
<span class="n">melbourne_model</span> <span class="o">=</span> <span class="n">DecisionTreeRegressor</span><span class="p">()</span>
<span class="c1"># Fit model</span>
<span class="n">melbourne_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">)</span>

<span class="c1"># get predicted prices on validation data</span>
<span class="n">val_predictions</span> <span class="o">=</span> <span class="n">melbourne_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">val_X</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">mean_absolute_error</span><span class="p">(</span><span class="n">val_y</span><span class="p">,</span> <span class="n">val_predictions</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>265806.91478373145
</pre>
</div>
</div>

</div>
</div>

</div>
    </div>

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell"><div id="sharing-control-portal-7" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Wow!" tabindex="0">Wow!<a class="anchor-link" href="https://www.kaggle.com/code/dansbecker/model-validation#Wow!" target="_self" rel=" noreferrer nofollow">¶</a></h1><p>Your mean absolute error for the in-sample data was about 500 dollars.  Out-of-sample it is more than 250,000 dollars.</p>
<p>This is the difference between a model that is almost exactly right, and one that is unusable for most practical purposes.  As a point of reference, the average home value in the validation data is 1.1 million dollars.  So the error in new data is about a quarter of the average home value.</p>
<p>There are many ways to improve this model, such as experimenting to find better features or different model types.</p>

</div>
</div>
</div>
    </div>

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell"><div id="sharing-control-portal-8" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Your-Turn" tabindex="0">Your Turn<a class="anchor-link" href="https://www.kaggle.com/code/dansbecker/model-validation#Your-Turn" target="_self" rel=" noreferrer nofollow">¶</a></h1><p>Before we look at improving this model, try <strong><a href="https://www.kaggle.com/kernels/fork/1259097">Model Validation</a></strong> for yourself.</p>

</div>
</div>
</div>
    </div>

    <div class="" role="section" tabindex="0">
        
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell"><div id="sharing-control-portal-9" class="sharing-control-portal"><div class="sc-bhjgvs eSYDTA cell-sharing-control"><div class="sc-irEpRR fEsHfI"><button aria-label="Copy cell link" title="Copy cell link" class="sc-pFPEP bgIjeN google-symbols notranslate">link</button><button aria-label="Embed cell" title="Embed cell" class="sc-pFPEP bgIjeN google-symbols notranslate">code</button></div></div></div>
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<p><em>Have questions or comments? Visit the <a href="https://www.kaggle.com/learn/intro-to-machine-learning/discussion">course discussion forum</a> to chat with other learners.</em></p>

</div>
</div>
</div>
    </div>

    </div>
