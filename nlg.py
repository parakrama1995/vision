# nlg.py
import random
import datetime
from py4j_server import launch_py4j_server
from py4j.java_gateway import java_import
package com.google.cloud.vision.samples.landmarkdetection;

// [START import_libraries]
import com.google.api.client.googleapis.auth.oauth2.GoogleCredential;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.services.vision.v1.Vision;
import com.google.api.services.vision.v1.VisionScopes;
import com.google.api.services.vision.v1.model.AnnotateImageRequest;
import com.google.api.services.vision.v1.model.AnnotateImageResponse;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesRequest;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesResponse;
import com.google.api.services.vision.v1.model.EntityAnnotation;
import com.google.api.services.vision.v1.model.Feature;
import com.google.api.services.vision.v1.model.Image;
import com.google.api.services.vision.v1.model.ImageSource;
import com.google.common.collect.ImmutableList;


gateway = launch_py4j_server()

# Import the SimpleNLG classes
java_import(gateway.jvm, "simplenlg.features.*")
java_import(gateway.jvm, "simplenlg.realiser.*")

# Define aliases so that we don't have to use the gateway.jvm prefix.
NPPhraseSpec = gateway.jvm.NPPhraseSpec
PPPhraseSpec = gateway.jvm.PPPhraseSpec
SPhraseSpec = gateway.jvm.SPhraseSpec
InterrogativeType = gateway.jvm.InterrogativeType
Realiser = gateway.jvm.Realiser
TextSpec = gateway.jvm.TextSpec
Tense = gateway.jvm.Tense
Form = gateway.jvm.Form


class NLG(object):
    """
    Used to generate natural language. Most of these sections are hard coded. However, some use simpleNLG which is
    used to string together verbs and nouns.
    """
    def __init__(self, user_name=None):
        self.user_name = user_name

        # make random more random by seeding with time
        random.seed(datetime.datetime.now())

    def acknowledge(self):

        user_name = self.user_name
        if user_name is None:
            user_name = ""

        simple_acknoledgement = [
            "Yes?",
            "What can I do for you?",
            "How can I help?"
        ]

        personal_acknowledgement = [
            "How can I help you today, %s" % user_name,
            "How can I help you, %s" % user_name,
            "What can I do for you, %s" % user_name,
            "Hi %s, what can I do for you?" % user_name,
            "Hey %s, what can I do for you?" % user_name
        ]

        choice = 0
        if self.user_name is not None:
            choice = random.randint(0, 2)
        else:
            choice = random.randint(0,1)

        ret_phrase = ""

        if choice == 0:
            ret_phrase = random.choice(simple_acknoledgement)
        elif choice == 1:
            date = datetime.datetime.now()
            ret_phrase = "Good %s. What can I do for you?" % self.time_of_day(date)
        else:
            ret_phrase = random.choice(personal_acknowledgement)

        return ret_phrase

    def searching(self):
        searching_phrases = [
            "I'll see what I can find"
        ]

        return random.choice(searching_phrases)

    def snow_white(self):

        phrases = [
            "You are",
            "You",
            "You are, of course"
        ]

        return random.choice(phrases)
    public static void main(String[] args) throws IOException, GeneralSecurityException {
    if (args.length != 1) {
      System.err.println("Usage:");
      System.err.printf("\tjava %s gs://<bucket_name>/<object_name>\n",
          DetectLandmark.class.getCanonicalName());
      System.exit(1);
    } else if (!args[0].toLowerCase().startsWith("gs://")) {
      System.err.println("Google Cloud Storage url must start with 'gs://'.");
      System.exit(1);
    }

    DetectLandmark app = new DetectLandmark(getVisionService());
    List<EntityAnnotation> landmarks = app.identifyLandmark(args[0], MAX_RESULTS);
    System.out.printf("Found %d landmark%s\n", landmarks.size(), landmarks.size() == 1 ? "" : "s");
    for (EntityAnnotation annotation : landmarks) {
      System.out.printf("\t%s\n", annotation.getDescription());
    }
  }
  // [END run_application]

  // [START authenticate]
  /**
   * Connects to the Vision API using Application Default Credentials.
   */
  public static Vision getVisionService() throws IOException, GeneralSecurityException {
    GoogleCredential credential =
        GoogleCredential.getApplicationDefault().createScoped(VisionScopes.all());
    JsonFactory jsonFactory = JacksonFactory.getDefaultInstance();
    return new Vision.Builder(GoogleNetHttpTransport.newTrustedTransport(), jsonFactory, credential)
            .setApplicationName(APPLICATION_NAME)
            .build();
  }
  // [END authenticate]

  // [START detect_gcs_object]
  private final Vision vision;

  /**
   * Constructs a {@link DetectLandmark} which connects to the Vision API.
   */
  public DetectLandmark(Vision vision) {
    this.vision = vision;
  }

  /**
   * Gets up to {@code maxResults} landmarks for an image stored at {@code uri}.
   */
  public List<EntityAnnotation> identifyLandmark(String uri, int maxResults) throws IOException {
    AnnotateImageRequest request =
        new AnnotateImageRequest()
            .setImage(new Image().setSource(
                new ImageSource().setGcsImageUri(uri)))
            .setFeatures(ImmutableList.of(
                new Feature()
                    .setType("LANDMARK_DETECTION")
                    .setMaxResults(maxResults)));
    Vision.Images.Annotate annotate =
        vision.images()
            .annotate(new BatchAnnotateImagesRequest().setRequests(ImmutableList.of(request)));
    // Due to a bug: requests to Vision API containing large images fail when GZipped.
    annotate.setDisableGZipContent(true);

    BatchAnnotateImagesResponse batchResponse = annotate.execute();
    assert batchResponse.getResponses().size() == 1;
    AnnotateImageResponse response = batchResponse.getResponses().get(0);
    if (response.getLandmarkAnnotations() == null) {
      throw new IOException(
          response.getError() != null
              ? response.getError().getMessage()
              : "Unknown error getting image annotations");
    }
    return response.getLandmarkAnnotations();
  }
  // [END detect_gcs_object]
}
def error_measure(predictions, labels):
    return np.sum(np.power(predictions - labels, 2)) / (2 * predictions.shape[0])


if __name__ == '__main__':
    train_dataset, train_labels = load_data()
    test_dataset, _ = load_data(test=True)

    # Generate a validation set.
    validation_dataset = train_dataset[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_dataset = train_dataset[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    train_size = train_labels.shape[0]
    print("train size is %d" % train_size)

    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))

    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    fc1_weights = tf.Variable(  # fully connected, depth 512.
                                tf.truncated_normal(
                                    [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                    stddev=0.1,
                                    seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    fc2_weights = tf.Variable(  # fully connected, depth 512.
                                tf.truncated_normal(
                                    [512, 512],
                                    stddev=0.1,
                                    seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    fc3_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # conv = tf.Print(conv, [conv], "conv1: ", summarize=10)

        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # relu = tf.Print(relu, [relu], "relu1: ", summarize=10)

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # pool = tf.Print(pool, [pool], "pool1: ", summarize=10)

        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # conv = tf.Print(conv, [conv], "conv2: ", summarize=10)

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        # relu = tf.Print(relu, [relu], "relu2: ", summarize=10)

        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # pool = tf.Print(pool, [pool], "pool2: ", summarize=10)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # reshape = tf.Print(reshape, [reshape], "reshape: ", summarize=10)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # hidden = tf.Print(hidden, [hidden], "hidden1: ", summarize=10)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)
        # hidden = tf.Print(hidden, [hidden], "hidden2: ", summarize=10)

        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        # return tf.nn.tanh(tf.matmul(hidden, fc3_weights) + fc3_biases)
        return tf.matmul(hidden, fc3_weights) + fc3_biases

    train_prediction = model(train_data_node, True)

    # Minimize the squared errors
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_prediction - train_labels_node), 1))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                    tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
    # Add the regularization term to the loss.
    loss += 1e-7 * regularizers

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = model(eval_data_node)

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    global_step = tf.Variable(0, trainable=False)

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        1e-3,                      # Base learning rate.
        global_step * BATCH_SIZE,  # Current index into the dataset.
        train_size,                # Decay step.
        0.95,                      # Decay rate.
        staircase=True)

    # train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    # train_step = tf.train.MomentumOptimizer(1e-4, 0.95).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    loss_train_record = list() # np.zeros(n_epoch)
    loss_valid_record = list() # np.zeros(n_epoch)
    start_time = time.gmtime()

    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0

    current_epoch = 0

    while current_epoch < NUM_EPOCHS:
        # Shuffle data
        shuffled_index = np.arange(train_size)
        np.random.shuffle(shuffled_index)
        train_dataset = train_dataset[shuffled_index]
        train_labels = train_labels[shuffled_index]

        for step in xrange(train_size / BATCH_SIZE):
            offset = step * BATCH_SIZE
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            _, loss_train, current_learning_rate = sess.run([train_step, loss, learning_rate], feed_dict=feed_dict)

        # After one epoch, make validation
        eval_result = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)
        loss_valid = error_measure(eval_result, validation_labels)

        print 'Epoch %04d, train loss %.8f, validation loss %.8f, train/validation %0.8f, learning rate %0.8f' % (
            current_epoch,
            loss_train, loss_valid,
            loss_train / loss_valid,
            current_learning_rate
        )
        loss_train_record.append(np.log10(loss_train))
        loss_valid_record.append(np.log10(loss_valid))
        sys.stdout.flush()

        if loss_valid < best_valid:
            best_valid = loss_valid
            best_valid_epoch = current_epoch
        elif best_valid_epoch + EARLY_STOP_PATIENCE < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
            break

        current_epoch += 1

    print('train finish')
    end_time = time.gmtime()
    print time.strftime('%H:%M:%S', start_time)
    print time.strftime('%H:%M:%S', end_time)

    generate_submission(test_dataset, sess, eval_prediction, eval_data_node)

    # Show an example of comparison
    i = 0
    img = validation_dataset[i]
    lab_y = validation_labels[i]
    lab_p = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)[0]
    plot_sample(img, lab_p, lab_y)

    plot_learning_curve(loss_train_record, loss_valid_record)
    def user_status(self, type='positive', attribute=None):

        ret_phrase = ""

        positive_complements = [
            "good",
            "nice",
            "great",
            "perfect",
            "Beautiful"
        ]

        negative_complements = [
            "bad",
            "terrible"
        ]

        moderate_complements = [
            "alright",
            "okay"
        ]

        complement_choice = positive_complements
        if type == 'negative':
            complement_choice = negative_complements
        elif type == 'moderate':
            complement_choice = moderate_complements

        if attribute is None:
            ret_phrase = "You look %s" % random.choice(complement_choice)
        else:
            ret_phrase = self.generate('none', {'subject': "Your %s" % attribute, 'verb': 'look %s' % random.choice(complement_choice)}, "present")

        return ret_phrase

    def personal_status(self, status_type=None):
        positive_status=[
            "I'm doing well",
            "Great, thanks for asking",
            "I'm doing great"
        ]

        negative_status = [
            "I'm not doing well",
            "I'm feeling terrible",
            "I'm not doing well today",
            "I could be much better"
        ]

        moderate_status = [
            "I'm doing alright",
            "I'm okay",
            "I could be better",
            "I'm alright"
        ]

        if status_type == 'negative':
            return random.choice(negative_status)
        elif status_type == 'moderate':
            return random.choice(moderate_status)

        return random.choice(positive_status)

    def joke(self):
        jokes = [
            "Artificial intelligence is no match for natural stupidity.",
            "This morning I made a mistake and poured milk over my breakfast instead of oil, and it rusted before I could eat it.",
            "An Englishman, an Irishman and a Scotsman walk into a bar. The bartender turns to them, takes one look, and says, \"What is this - some kind of joke?\"",
            "What's an onomatopoeia? Just what it sounds like!",
            "Why did the elephant cross the road? Because the chicken retired.",
            "Today a man knocked on my door and asked for a small donation towards the local swimming pool. I gave him a glass of water.",
            "A recent study has found that women who carry a little extra weight live longer than the men who mention it.",
            "I can totally keep secrets. It's the people I tell them to that can't.",
            "My therapist says I have a preoccupation with vengeance. We'll see about that.",
            "Money talks ...but all mine ever says is good-bye.",
            "I started out with nothing, and I still have most of it.",
            "I used to think I was indecisive, but now I'm not too sure.",
            "I named my hard drive dat ass so once a month my computer asks if I want to 'back dat ass up'.",
            "A clean house is the sign of a broken computer.",
            "My favorite mythical creature? The honest politician.",
            "Regular naps prevent old age, especially if you take them while driving.",
            "For maximum attention, nothing beats a good mistake.",
            "Take my advice. I'm not using it."
        ]

        return random.choice(jokes)

    def news(self, tense):

        headlines_nouns = [
            "stories",
            "articles",
            "headlines",
        ]

        headlines_adjectives = [
            ["these"],
            ["some"],
            ["a", "few"],
            ["a", "couple"]
        ]

        headlines_prepmodifiers = [
            "you"
        ]

        choice = random.randint(0, 1)

        if choice == 1:
            ret_phrase = self.generate('none', {'subject': "I", 'object': random.choice(headlines_nouns), 'verb': 'find', 'objmodifiers': random.choice(headlines_adjectives), 'preposition': 'for', 'prepmodifiers': [random.choice(headlines_prepmodifiers)]}, tense)
        else:
            ret_phrase = self.generate('none', {'subject': "I", 'object': random.choice(headlines_nouns), 'verb': 'find', 'objmodifiers': random.choice(headlines_adjectives)}, tense)

        return ret_phrase

    def article_interest(self, article_titles):
        ret_phrase = None

        if random.randint(0,2) == 0: # 1 in 3 chance the bot will express interest in a random article
            if article_titles is not None:
                article = random.choice(article_titles)
                article = article.rsplit('-', 1)[0]
                ret_phrase = "%s sounds particularly interesting" % article

        return ret_phrase

    def insult(self):
        return "That's not very nice. Talk to me again when you have fixed your attitude"

    def greet(self):
        """
        Creates a greeting phrase.
        :return:
        """

        greeting_words = [
            "Hi",
            "Hey",
            "Hello"
        ]

        goofy_greetings = [
            "what's up?",
            "howdy",
            "what's crackin'?",
            "top of the morning to ya"
        ]

        choice = random.randint(0,4)
        ret_phrase = ""

        if (choice == 0) or (choice == 3): # time related
            ret_phrase = "Good %s" % self.time_of_day(datetime.datetime.now())
            if self.user_name is not None:
                if random.randint(0, 1) == 0:
                    ret_phrase = "%s %s" % (ret_phrase, self.user_name)
        elif (choice == 1) or (choice == 4): # standard greeting
            ret_phrase = random.choice(greeting_words)
            if self.user_name is not None:
                if random.randint(0, 1) == 0:
                    ret_phrase = "%s %s" % (ret_phrase, self.user_name)
        elif choice == 2: # goofy greeting
            ret_phrase = random.choice(goofy_greetings)

        return ret_phrase

    def weather(self, temperature, date, tense):
        """
        Generates a statement about the current weather.
        :param temperature:
        :param date:
        :param tense:
        :return:
        """

        ret_phrase = self.generate('none', {'subject':"the temperature", 'object': "%d degrees" % temperature, 'verb': 'is', 'adverbs': ["%s" % self.time_of_day(date, with_adjective=True)]}, tense)
        return ret_phrase

    def forecast(self, forecast_obj):

        ret_phrase = ""
        forecast = ""

        if forecast_obj.get("forecast") is None:
            return ret_phrase
        else:
            forecast = forecast_obj.get("forecast")

        forecast_current = [
            "Currently, it's",
            "Right now, it's",
            "At the moment, it's",
            "It's",
            "It is"
        ]

        forecast_hourly = [
            "It's",
            "It will be",
            "Looks like it will be"
        ]

        forecast_daily = [
            ""
        ]

        if forecast_obj.get('forecast_type') == "current":
            ret_phrase = "%s %s" % (random.choice(forecast_current), forecast)
        elif forecast_obj.get('forecast_type') == "hourly":
            ret_phrase = "%s %s" % (random.choice(forecast_hourly), forecast)
        elif forecast_obj.get('forecast_type') == "daily":
            ret_phrase = "%s %s" % (random.choice(forecast_daily), forecast)

        return ret_phrase

    def appreciation(self):
        phrases = [
            "No problem!",
            "Any time",
            "You are welcome",
            "You're welcome",
            "Sure, no problem",
            "Of course",
            "Don't mention it",
            "Don't worry about it"
        ]

        return random.choice(phrases)

    def holiday(self, holiday_name):
        phrases = [
            "",
            "Looks like the next holiday is ",
            "The next important day is "
        ]

        return "%s%s" % (random.choice(phrases), holiday_name)

    def meaning_of_life(self):
        phrases = [
            "42",
            "The meaning of life, the universe, and everything else is 42"
        ]

        return random.choice(phrases)

    def name(self):
        return self.user_name

    def time_of_day(self, date, with_adjective=False):
        ret_phrase = ""
        if date.hour < 10:
            ret_phrase = "morning"
            if with_adjective:
                ret_phrase = "%s %s" % ("this", ret_phrase)
        elif (date.hour >= 10) and (date.hour < 18):
            ret_phrase = "afternoon"
            if with_adjective:
                ret_phrase = "%s %s" % ("this", ret_phrase)
        elif date.hour >= 18:
            ret_phrase = "evening"
            if with_adjective:
                ret_phrase = "%s %s" % ("this", ret_phrase)

        return ret_phrase

    def generate(self, utter_type, keywords, tense=None):
        """
        Input: a type of inquiry to create and a dictionary of keywords.
        Types of inquiries include 'what', 'who', 'where', 'why', 'how',
        and 'yes/no' questions. Alternatively, 'none' can be specified to
        generate a declarative statement.

        The dictionary is essentially divided into three core parts: the
        subject, the verb, and the object. Modifiers can be specified to these
        parts (adverbs, adjectives, etc). Additionally, an optional
        prepositional phrase can be specified.

        Example:

        nlg = NaturalLanguageGenerator(logging.getLogger())
        words = {'subject': 'you',
                 'verb': 'prefer',
                 'object': 'recipes',
                 'preposition': 'that contains',
                 'objmodifiers': ['Thai'],
                 'prepmodifiers': ['potatoes', 'celery', 'carrots'],
                 'adverbs': ['confidently'],
        }
        NPPhraseSpec = gateway.jvm.NPPhraseSpec
        PPPhraseSpec = gateway.jvm.PPPhraseSpec
        SPhraseSpec = gateway.jvm.SPhraseSpec
        InterrogativeType = gateway.jvm.InterrogativeType
        Realiser = gateway.jvm.Realiser
        TextSpec = gateway.jvm.TextSpec
        Tense = gateway.jvm.Tense
        Form = gateway.jvm.Form


        nlg.generate('yes_no', words)
        u'Do you confidently prefer Thai recipes that contains potatoes, celery and carrots?'
        nlg.generate('how', words)
        u'How do you confidently prefer Thai recipes that contains potatoes, celery and carrots?'
        """
        utterance = SPhraseSpec()
        subject = NPPhraseSpec(keywords['subject'])
        target = None
        if 'object' in keywords:
            target = NPPhraseSpec(keywords['object'])
        preposition = PPPhraseSpec()

        if 'preposition' in keywords:
            preposition.setPreposition(keywords['preposition'])

        if 'prepmodifiers' in keywords:
            for modifier in keywords['prepmodifiers']:
                preposition.addComplement(modifier)

        if 'submodifiers' in keywords:
            for modifier in keywords['submodifiers']:
                subject.addModifier(modifier)

        if 'objmodifiers' in keywords:
            for modifier in keywords['objmodifiers']:
                target.addModifier(modifier)

        if utter_type.lower() == 'yes_no':
            utterance.setInterrogative(InterrogativeType.YES_NO)
        elif utter_type.lower() == 'how':
            utterance.setInterrogative(InterrogativeType.HOW)
        elif utter_type.lower() == 'what':
            utterance.setInterrogative(InterrogativeType.WHAT)
        elif utter_type.lower() == 'where':
            utterance.setInterrogative(InterrogativeType.WHERE)
        elif utter_type.lower() == 'who':
            utterance.setInterrogative(InterrogativeType.WHO)
        elif utter_type.lower() == 'why':
            utterance.setInterrogative(InterrogativeType.WHY)

        if target is not None:
            target.addModifier(preposition)
        utterance.setSubject(subject)
        utterance.setVerb(keywords['verb'])
        if 'adverbs' in keywords:
            for modifier in keywords['adverbs']:
                utterance.addModifier(modifier)
        if target is not None:
            utterance.addComplement(target)

        if tense.lower() == 'future':
            utterance.setTense(Tense.FUTURE)
        elif tense.lower() == 'past':
            utterance.setTense(Tense.PAST)

        realiser = Realiser()
        output = realiser.realiseDocument(utterance).strip()
        return output
    
    public static void main(String[] args) throws IOException, GeneralSecurityException {
    if (args.length != 1) {
      System.err.println("Usage:");
      System.err.printf("\tjava %s gs://<bucket_name>/<object_name>\n",
          DetectLandmark.class.getCanonicalName());
      System.exit(1);
    } else if (!args[0].toLowerCase().startsWith("gs://")) {
      System.err.println("Google Cloud Storage url must start with 'gs://'.");
      System.exit(1);
    }
