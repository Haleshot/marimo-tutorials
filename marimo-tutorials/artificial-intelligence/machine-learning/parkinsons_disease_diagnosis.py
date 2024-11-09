import marimo

__generated_with = "0.8.19"
app = marimo.App(
    app_title="Exploring Parkinson's Disease Diagnosis",
    css_file="",
)


@app.cell(hide_code=True)
def __(header_widget):
    header_widget
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        f"""
        {mo.accordion({
            "Abstract": mo.md("""Parkinson's Disease (PD) is a neurodegenerative disorder impacting movement and speech. Early and accurate diagnosis is crucial for timely intervention. This study investigates the potential of machine learning to analyze voice recordings for PD detection. We simulate a dataset using a Variational Autoencoder (VAE) to address potential limitations of real-world datasets. Subsequently, we explore various regression models for the task and compare their performance using cross-validation techniques. This report outlines the methodology, analyzes the results, and provides insights into selecting the most suitable model for PD diagnosis based on voice data."""),

        "Introduction": mo.md("""Parkinson's Disease is a progressive neurological disorder characterized by a decline in dopamine production within the brain. This leads to motor impairments such as tremors, rigidity, and slowness of movement. Speech is also commonly affected, with symptoms including slurred speech (dysarthria), reduced volume (hypophonia), and a monotonous pitch. Early and accurate diagnosis of PD is vital for optimizing treatment strategies and improving patient outcomes. Traditional diagnosis relies on a clinician's assessment of a patient's medical history and motor skills. However, the absence of a definitive lab test makes early diagnosis challenging.

        This study explores the potential of machine learning to analyze voice recordings for PD detection. Voice offers a non-invasive and accessible data source for screening purposes. By leveraging machine learning algorithms to identify characteristic vocal features in PD patients, we aim to develop an effective screening tool to supplement traditional diagnostic methods. This report details the simulation of a voice recordings dataset using a VAE, the application of various regression models for the classification task, and a comparative analysis of their performance. The findings will provide valuable insights into the feasibility of using voice-based machine learning for PD diagnosis."""),
        "About Dataset": mo.md("""

        This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to the "status" column which is set to 0 for healthy and 1 for PD.

        The data is in ASCII CSV format. The rows of the CSV file contain an instance corresponding to one voice recording. There are around six recordings per patient, the name of the patient is identified in the first column.For further information or to pass on comments, please contact Max Little (little '@' robots.ox.ac.uk).

        Further details are contained in the following reference 

        Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', IEEE Transactions on Biomedical Engineering (to appear).

        """),

        "Column Information": mo.md("""

        - name - ASCII subject name and recording number
        - MDVP:Fo(Hz) - Average vocal fundamental frequency
        - MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
        - MDVP:Flo(Hz) - Minimum vocal fundamental frequency
        - MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP - Several measures of variation in fundamental frequency
        - MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
        - NHR, HNR - Two measures of the ratio of noise to tonal components in the voice
        - status - The health status of the subject (one) - Parkinson's, (zero) - healthy
        - RPDE, D2 - Two nonlinear dynamical complexity measures
        - DFA - Signal fractal scaling exponent
        - spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation
        """),
    })}
        <br>
        You can explore the dataset in the data explorer below.
        """
    )
    return


@app.cell(hide_code=True)
def __(data, mo):
    mo.ui.data_explorer(data.copy())
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We standardize the features and create PyTorch Dataset & DataLoeader""")
    return


@app.cell(hide_code=True)
def __(DATA_PATH, DataLoader, Dataset, StandardScaler, np, pd):
    # Load the dataset
    # data = oversample_data("parkinsons.csv", "status")
    data = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = data.drop(["name", "status"], axis=1).values
    y = data["status"].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine features and target
    data_combined = np.hstack((X_scaled, y.reshape(-1, 1)))


    class ParkinsonsDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    # Create DataLoader
    dataset = ParkinsonsDataset(data_combined)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return (
        ParkinsonsDataset,
        X,
        X_scaled,
        data,
        data_combined,
        dataloader,
        dataset,
        scaler,
        y,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Data Simulation with Conditional Variational Autoencoder (CVAE)

        ---

        > **If you are familiar with VAR and CVAE, feel free to jump to [next section](#lasso) or our [implementation](#cvae-implementation) of utilizing CVAE to simulate data.**

        <h2 align="center">Rationale for Data Simulation</h2>

        Data simulation is crucial for augmenting existing datasets, particularly when working with limited or imbalanced data. In the context of Parkinson's Disease (PD) diagnosis, generating synthetic voice recordings helps create a more robust dataset, ensuring better generalization of machine learning models. By simulating additional data, we can enhance model training and improve the reliability of predictions.

        <h2 align="center">VAE Model Architecture and Training</h2>

        Variational Autoencoders (VAEs) are a powerful generative model that learns to represent data in a latent space, enabling the generation of new, synthetic data samples. The VAE consists of an encoder, which maps input data to a latent space, and a decoder, which reconstructs the data from this latent space. We detail the VAE architecture, training process, and the loss function that balances reconstruction accuracy and latent space regularization.

        #### Encoder

        The encoder maps input data \( \mathbf{x} \) to a latent space \( \mathbf{z} \). It outputs two vectors: the mean \( \boldsymbol{\mu} \) and the logarithm of the variance \( \boldsymbol{\log \sigma^2} \) of the latent variables. The latent variables are sampled from a Gaussian distribution:

        \[ \mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \]

        The encoder can be represented as:

        \[ \boldsymbol{\mu}, \boldsymbol{\log \sigma^2} = \text{Encoder}(\mathbf{x}) \]

        #### Reparameterization Trick

        To enable backpropagation through the sampling step, we use the reparameterization trick:

        \[ \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \mathbf{\epsilon} \]

        \[ \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \]

        #### Decoder

        The decoder maps the latent variables \( \mathbf{z} \) back to the data space, reconstructing the input data:

        \[ \hat{\mathbf{x}} = \text{Decoder}(\mathbf{z}) \]

        #### Loss Function

        The VAE is trained using a loss function that combines the reconstruction loss and the Kullback-Leibler (KL) divergence:

        1. **Reconstruction Loss**: Measures how well the decoder reconstructs the input data. Commonly used reconstruction loss is the binary cross-entropy (BCE) or mean squared error (MSE):

        \[ \text{Reconstruction Loss} = \text{BCE}(\mathbf{x}, \hat{\mathbf{x}}) \]

        2. **KL Divergence**: Regularizes the latent space to follow a standard normal distribution:

        \[ \text{KL Divergence} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right) \]

        Combining both, the total VAE loss function is:

        \[ \mathcal{L} = \text{Reconstruction Loss} + \beta \cdot \text{KL Divergence} \]

        where \( \beta \) is a weight that balances the two terms.

        #### Training

        The training process involves minimizing the total loss function using stochastic gradient descent (SGD) or its variants. The steps are as follows:

        1. **Forward Pass**: Pass the input data through the encoder to obtain \( \boldsymbol{\mu} \) and \( \boldsymbol{\log \sigma^2} \).
        2. **Reparameterization**: Sample \( \mathbf{z} \) using the reparameterization trick.
        3. **Decoder Pass**: Pass \( \mathbf{z} \) through the decoder to get \( \hat{\mathbf{x}} \).
        4. **Compute Loss**: Calculate the reconstruction loss and KL divergence, then compute the total loss.
        5. **Backpropagation**: Compute gradients and update the model parameters using an optimizer like Adam.

        The VAE model architecture and training procedure enable the generation of high-quality synthetic data, which can be used to augment the original dataset and improve machine learning model performance.


        <h2 align="center">Conditional Variational Autoencoder (CVAE)</h2>

        Conditional Variational Autoencoder (CVAE) is an extension of the standard Variational Autoencoder (VAE) that incorporates conditional information into the generation process. In a traditional VAE, the latent space is learned from the input data without considering any specific attributes or labels associated with the data. However, in many practical applications, it's desirable to generate data conditioned on certain attributes or labels.

        #### Architecture
        The architecture of a CVAE is similar to that of a standard VAE, consisting of an encoder network, a decoder network, and a sampling mechanism in the latent space. The key difference lies in the addition of conditional information to both the encoder and decoder.

        - **Encoder**: The encoder takes both the input data and the conditional information as inputs and outputs the parameters of the latent distribution. It learns to map the input data and conditional information to the parameters of a multivariate Gaussian distribution in the latent space.

        - **Decoder**: The decoder takes samples from the latent space, along with the conditional information, and reconstructs the input data. It learns to map samples from the latent space and conditional information to the output data space.

        #### Training
        The training objective of a CVAE is to maximize the evidence lower bound (ELBO), similar to a standard VAE. However, in a CVAE, the conditional information is incorporated into both the reconstruction loss and the KL divergence term of the ELBO.

        - **Reconstruction Loss**: Measures the difference between the input data and the reconstructed data. It ensures that the generated data closely matches the input data, conditioned on the provided attributes or labels.

        - **KL Divergence**: Measures the discrepancy between the learned latent distribution and the prior distribution. It encourages the latent space to follow a prior distribution, typically a standard Gaussian distribution, while taking into account the conditional information.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h2 id="cvae-implementation" align="center">Implementation of CVAE</h2>

        > Feel free to tweek anything
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""CVAE Architecture Implementaion""")
    return


@app.cell(hide_code=True)
def __(F, nn, torch):
    class Encoder(nn.Module):
        def __init__(self, input_dim, cond_dim, latent_dim, hidden_dim):
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
            self.fc21 = nn.Linear(hidden_dim, latent_dim)
            self.fc22 = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x, c):
            x = torch.cat((x, c), dim=1)
            h = F.relu(self.fc1(x))
            mu = self.fc21(h)
            logvar = self.fc22(h)
            return mu, logvar


    class Decoder(nn.Module):
        def __init__(self, latent_dim, cond_dim, output_dim, hidden_dim):
            super(Decoder, self).__init__()
            self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, z, c):
            z = torch.cat((z, c), dim=1)
            h = F.relu(self.fc1(z))
            x_recon = self.fc2(h)
            return x_recon


    class CVAE(nn.Module):
        def __init__(self, input_dim, cond_dim, latent_dim, hidden_dim=64):
            super(CVAE, self).__init__()
            self.encoder = Encoder(input_dim, cond_dim, latent_dim, hidden_dim)
            self.decoder = Decoder(latent_dim, cond_dim, input_dim, hidden_dim)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x, c):
            mu, logvar = self.encoder(x, c)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z, c)
            return x_recon, mu, logvar
    return CVAE, Decoder, Encoder


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Function to train CVAE, you can use any other otimizer from PyTorch, and tweet `num_epochs`.""")
    return


@app.cell(hide_code=True)
def __(mo):
    def train_cvae_model(
        model,
        dataloader,
        optimizer,
        loss_function,
        num_epochs=500,
        verbose: bool = True,
    ):
        """
        Training loop for the CVAE model.

        Args:
            model: The model to be trained.
            dataloader: DataLoader for the training data.
            optimizer: Optimizer for updating model parameters.
            loss_function: Loss function to calculate the training loss.
            num_epochs (int): Number of training epochs. Default is 500.

        Returns:
            list: List of training losses for each epoch.
        """
        train_loss = []  # Initialize an empty list to store the training loss

        for epoch in mo.status.progress_bar(range(num_epochs)):
            model.train()
            total_loss = 0
            for data in dataloader:
                data = data.float()
                x = data[:, :-1]
                c = data[:, -1].unsqueeze(1)

                optimizer.zero_grad()
                x_recon, mu, logvar = model(x, c)
                loss = loss_function(x_recon, x, mu, logvar)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            avg_loss = total_loss / len(dataloader.dataset)
            train_loss.append(avg_loss)  # Append the average loss to the list

        return train_loss
    return (train_cvae_model,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Train the Model, you can experiment with different `latent_dim`""")
    return


@app.cell(hide_code=True)
def __(CVAE, F, X_scaled, dataloader, optim, torch, train_cvae_model):
    # Define model, optimizer and loss function
    input_dim = X_scaled.shape[1]
    cond_dim = 1  # Binary condition (status)
    latent_dim = 10

    model = CVAE(input_dim, cond_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 300


    def loss_function(recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


    train_loss = train_cvae_model(
        model, dataloader, optimizer, loss_function, num_epochs
    )
    return (
        cond_dim,
        input_dim,
        latent_dim,
        loss_function,
        model,
        num_epochs,
        optimizer,
        train_loss,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.callout(
        mo.md(
            """
            Tip:

            You can find the implementation of a function in the **Explore variables** panel.

            For example, by typing "plot_train_loss", you can find in which cell this function is implemented and navigate to it.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The training loss is shown below""")
    return


@app.cell(hide_code=True)
def __(plot_train_loss, train_loss):
    # After the training loop, visualize the training loss
    plot_train_loss(train_loss)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Function to generate conditional synthetic data using a trained CVAE model.""")
    return


@app.cell(hide_code=True)
def __(np, random, torch):
    def generate_conditional_synthetic_data(
        model, num_samples, target_variable, latent_dim, num_classes, weights
    ):
        """
        Generate synthetic data using a trained CVAE model.

        Args:
            model (CVAE): Trained CVAE model.
            num_samples (int): Number of samples to generate.
            target_variable (list): List of potential values for the target variable.
            latent_dim (int): Dimension of the latent space.
            num_classes (int): Number of classes for the target variable.
            weights (list[float]): list of weight for each class

        Returns:
            torch.Tensor: Generated synthetic data.
        """
        model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim)
            y_values = random.choices(
                target_variable, weights=weights, k=num_samples
            )  # Randomly sample y values
            y = torch.tensor(y_values).unsqueeze(1)
            generated_data = model.decoder(z, y).detach().numpy()

            # Combine generated data and y values
            generated_data_with_y = np.hstack((generated_data, y.numpy()))

        return generated_data_with_y
    return (generate_conditional_synthetic_data,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Generate 500 conditional synthetic data using a trained CVAE model.""")
    return


@app.cell(hide_code=True)
def __(generate_conditional_synthetic_data, latent_dim, model):
    # Generate synthetic data
    synthetic_data = generate_conditional_synthetic_data(
        model, 500, (0, 1), latent_dim, 2, weights=(1 / 3, 1)
    )
    return (synthetic_data,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h2 align="center">Test the Quality of Synthetic Data</h2>

        Now, we want to test the quality of synthetic data.

        The logic is simple, we train a classifier to classify a datapoint as **true** or **synthetic**. If the quality of synthetic data is great, then the accuracy of this trained classifier should be around 50%, i.e., the distributions of true data and synthetic data are similar and the classifier fails to distinguish.

        First, let's implement some helper functions.
        """
    )
    return


@app.cell(hide_code=True)
def __(np):
    def label_data(original_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Label original data as 0 and synthetic data as 1.

        Parameters:
        original_data (np.ndarray): The original dataset.
        synthetic_data (np.ndarray): The synthetic dataset.

        Returns:
        labeled_original_data (np.ndarray): Original data with label 0.
        labeled_synthetic_data (np.ndarray): Synthetic data with label 1.
        """
        labeled_original_data = np.hstack(
            (original_data, np.zeros((original_data.shape[0], 1)))
        )
        labeled_synthetic_data = np.hstack(
            (synthetic_data, np.ones((synthetic_data.shape[0], 1)))
        )
        return labeled_original_data, labeled_synthetic_data
    return (label_data,)


@app.cell(hide_code=True)
def __(np, train_test_split):
    def split_data(
        labeled_original_data: np.ndarray,
        labeled_synthetic_data: np.ndarray,
        test_size: float = 0.5,
    ):
        """
        Split original and synthetic data into train and test sets.

        Parameters:
        labeled_original_data (np.ndarray): Labeled original dataset.
        labeled_synthetic_data (np.ndarray): Labeled synthetic dataset.
        test_size (float): Proportion of the dataset to include in the test split.

        Returns:
        X_train, X_test, y_train, y_test (np.ndarray): Train and test sets.
        """
        original_train, original_test = train_test_split(
            labeled_original_data, test_size=test_size, random_state=42
        )
        synthetic_train, synthetic_test = train_test_split(
            labeled_synthetic_data, test_size=test_size, random_state=42
        )

        train_data = np.vstack((original_train, synthetic_train))
        test_data = np.vstack((original_test, synthetic_test))

        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]

        return X_train, X_test, y_train, y_test
    return (split_data,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We use **RandomForestClassifier** for classification here.""")
    return


@app.cell(hide_code=True)
def __(RandomForestClassifier, accuracy_score, np):
    def train_and_evaluate_classifier(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Train a classifier on the training set and evaluate it on the test set.

        Parameters:
        X_train, X_test (np.ndarray): Training and testing features.
        y_train, y_test (np.ndarray): Training and testing labels.

        Returns:
        accuracy (float): Accuracy of the classifier on the test set.
        """
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy
    return (train_and_evaluate_classifier,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        We repeat the evaluation 10 times and take the median value.

        > we always prefer **median** to **mean**
        """
    )
    return


@app.cell(hide_code=True)
def __(pd, split_data, statistics, train_and_evaluate_classifier):
    def repeated_evaluation(
        labeled_original_data: pd.DataFrame,
        labeled_synthetic_data: pd.DataFrame,
        n_repeats: int = 10,
    ):
        """
        Repeat the train-test split and classifier evaluation for a specified number of times and return the median accuracy.

        Parameters:
        labeled_original_data, labeled_synthetic_data (pd.DataFrame): Labeled original and synthetic datasets.
        n_repeats (int): Number of repetitions for train-test split and evaluation.

        Returns:
        median_accuracy (float): Median accuracy over the repetitions.
        """
        accuracies = [
            train_and_evaluate_classifier(
                *split_data(labeled_original_data, labeled_synthetic_data)
            )
            for _ in range(n_repeats)
        ]

        median_accuracy = statistics.median(accuracies)
        return median_accuracy
    return (repeated_evaluation,)


@app.cell(hide_code=True)
def __(data_combined, label_data, mo, repeated_evaluation, synthetic_data):
    labeled_original_data, labeled_synthetic_data = label_data(
        data_combined, synthetic_data
    )
    median_accuracy = repeated_evaluation(
        labeled_original_data, labeled_synthetic_data
    )
    with mo.redirect_stdout():
        print(f"Median accuracy over 10 repetitions: {median_accuracy:.2f}")
    return labeled_original_data, labeled_synthetic_data, median_accuracy


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        It seems the quality of our synthetic data isn't that good.

        > We also tested with other values of `latent_dim`, all of them get similar results.

        Generally, we want to illustrate a possible method of doing machine learning tasks, albeit its behaviour vary in different datasets.

        Anyway, let's move on.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo, np, plot_lasso_coefficients, synthetic_data):
    mo.md(
        f"""
        <h1 id="lasso">Using Lasso Model to Visualize Predictor Importance</h1>

        > there are other ways to explor predictor importance, for example, train a RandomForest model and view the importance of each predictor.

        In this section, we use Lasso Model to visualize the importance of each predictor, the pipeline is as follows:

        1. **Standardize the Data:** Ensure that all predictors (features) are standardized to have zero mean and unit variance. This is important because Lasso regression is sensitive to the scale of the input data.
        2. **Fit Lasso Model with Varying Alpha Values:** Train multiple Lasso regression models using a range of alpha values (regularization strengths). The alpha parameter controls the degree of sparsity in the model coefficients. Smaller alpha values lead to less regularization, while larger alpha values increase the regularization, potentially shrinking more coefficients to zero.
        3. **Record Coefficients:** For each alpha value, record the coefficients of the predictors. This will give us a series of coefficients for each predictor across different alpha values.
        4. **Plot the Coefficients:** Plot the coefficients of each predictor against the corresponding alpha values on a single graph. This will help visualize how the importance of each predictor changes with varying levels of regularization.

        The result is as follows: 

        {mo.ui.plotly(plot_lasso_coefficients(
        synthetic_data[:, :-1],
        synthetic_data[:, -1],
        np.logspace(-2, 0, 10),
    ))}

        Result Illustration:

        - X-axis: Represents the alpha values (log-scaled for better visualization).
        - Y-axis: Represents the coefficients of the predictors.
        - Lines: Each line on the graph represents the coefficient path of a single predictor as alpha varies.
        - Starting Point: At lower alpha values, most coefficients are non-zero, indicating that many predictors are being used by the model.
        - Increasing Alpha: As alpha increases, the coefficients of less important predictors shrink towards zero.
        - Significant Predictors: At a certain point, only a few predictors remain with coefficients significantly above zero. In this case, it is observed that only 4 predictors have coefficients significantly above zero at the starting point.

        The significance of few predictors inspire us to reduce the number of predictors.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h1>Reducing the Number of Predictors with an Autoencoder</h1>

        > there are other ways to do dimensionality reduction, for example, PCA.

        In this section, we use Autoencoder to reduce the number of predictors to 4.

        The pipeline is as follows: 

        1. **Build the Autoencoder:** Create an autoencoder model with an input layer corresponding to the number of original predictors, hidden layers to compress the data, and a bottleneck layer with 4 neurons to reduce the dimensionality to 4.
        2. **Train the Autoencoder:** Train the autoencoder using the original dataset, ensuring that the encoder learns to compress the data into the 4-dimensional space effectively.
        3. **Transform Data:** Use the trained encoder part of the autoencoder to transform the original high-dimensional data into the reduced 4-dimensional space.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Since we've illustrated the implementation of CVAE, we will be more concise here.""")
    return


@app.cell(hide_code=True)
def __(nn, np, optim, torch):
    # Define the Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    def train_autoencoder(
        model, data, epochs=100, batch_size=32, learning_rate=0.001
    ):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        data_tensor = torch.FloatTensor(data)
        dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        train_loss = []  # Initialize an empty list to store the training loss

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs, _ = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader.dataset)
            train_loss.append(avg_loss)  # Append the average loss to the list

        return train_loss


    def transform_data(model, data):
        """
        Transform the input data using the encoder part of the trained autoencoder.

        Args:
            model (Autoencoder): Trained autoencoder model.
            data (numpy.ndarray): Input data with the last column as the target variable.

        Returns:
            numpy.ndarray: Transformed data with the same target variable appended as the last column.
        """
        model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(
                data[:, :-1]
            )  # Exclude the target column for transformation
            encoded_data = model.encoder(data_tensor).numpy()

        # Append the target column back to the transformed data
        transformed_data_with_target = np.hstack(
            (encoded_data, data[:, -1].reshape(-1, 1))
        )
        return transformed_data_with_target
    return Autoencoder, train_autoencoder, transform_data


@app.cell(hide_code=True)
def __(Autoencoder, synthetic_data, train_autoencoder):
    autoencoder = Autoencoder(input_dim=synthetic_data.shape[1] - 1, latent_dim=4)

    # Train the Autoencoder
    autoencoder_train_loss = train_autoencoder(
        autoencoder,
        synthetic_data[:, :-1],
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
    )
    return autoencoder, autoencoder_train_loss


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The learning curve is as follows:""")
    return


@app.cell
def __(autoencoder_train_loss, plot_train_loss):
    plot_train_loss(autoencoder_train_loss)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Transform data with the trained AutoEncoder""")
    return


@app.cell(hide_code=True)
def __(autoencoder, synthetic_data, transform_data):
    transformed_synthetic_data = transform_data(autoencoder, synthetic_data)
    return (transformed_synthetic_data,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We formally create the DataFrame here.""")
    return


@app.cell(hide_code=True)
def __(pd, transformed_synthetic_data):
    df = pd.DataFrame(
        transformed_synthetic_data,
        columns=["AE-1", "AE-2", "AE-3", "AE-4", "status"],
    )
    return (df,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Next, we will do some EDA""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h1>Exploratory Data Analysis</h1>

        In this section, we do some EDA on the transformed data, the pipeline is as follows:


        **1. Feature Distribution Analysis:**

        - **Overlay Density Plots:** Visualize the distribution of each extracted feature for both classes using overlaid density plots. This reveals how the values are spread out for each class within a feature.
        - **Violin Plots:** Compare the distribution of each feature across classes using violin plots. This helps identify differences in shape, central tendency (median), and spread (interquartile range) between the classes for each feature.

        **2. Relationship Exploration:**

        - **Pairplot:** Create a pairplot to visualize the relationships between all three features simultaneously. This helps identify potential correlations or clusters within your data.
        - **Correlation Heatmap:** Calculate the correlation coefficients between the features and display them as a heatmap. This provides a quick overview of the linear relationships between the features.

        The results are displayed below:
        """
    )
    return


@app.cell(hide_code=True)
def __(create_distplot, transformed_synthetic_data):
    create_distplot(transformed_synthetic_data)
    return


@app.cell(hide_code=True)
def __(create_violin, transformed_synthetic_data):
    create_violin(transformed_synthetic_data)
    return


@app.cell(hide_code=True)
def __(create_pairplot, df):
    create_pairplot(df)
    return


@app.cell(hide_code=True)
def __(correlation_heatmap, df):
    correlation_heatmap(df)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Result Illustration:

        1. **Distinct Feature Distributions:** The observation that distributions of features, particularly `AE-1` and `AE-2, vary for the two classes is a positive sign. This suggests that the autoencoder has successfully captured features that differentiate status 0 and 1. These features can potentially be used to build an effective classifier.

        2. **Low Feature Correlation:** The low correlation between the features indicates that they likely capture different aspects of the data. This is generally desirable for classification tasks, as uncorrelated features provide more independent information about the data points.

        Now we will start model training.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <style>
            body {
          margin: 0;
          padding: 0;
        }

        .container {
          max-width: 1100px;
          margin: 2rem auto;
          padding: 1rem;
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .section {
          border-radius: 15px;
          padding: 2rem;
        }

        .header {
          text-align: center;
          margin-bottom: 1.5rem;
        }

        .header h1 {
          font-size: 2.5rem;
          text-shadow: 0 0 10px #76c7c0;
        }

        .model-cards {
          display: flex;
          flex-wrap: wrap;
          gap: 1.5rem;
          justify-content: space-between;
        }

        .model-card {
          padding: 1.5rem;
          border-radius: 12px;
          flex: 1 1 48%;
          box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
          transition: transform 0.3s ease-in-out;
        }

        .model-card:hover {
          transform: translateY(-10px);
        }

        .model-card h2 {
          font-size: 1.5rem;
          margin-bottom: 0.8rem;
        }

        .model-card p {
          line-height: 1.6;
        }

        .overall {
          padding: 1.5rem;
          border-radius: 12px;
          box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
          margin-top: 1.5rem;
        }

        .overall p {
          font-size: 1.2rem;
        }

        .heading {
          text-align: center;
          font-size: 2rem;
          text-shadow: 0 0 6px rgba(255, 136, 0, 0.5);
        }

        .cross-validation ol {
          padding-left: 1.5rem;
          line-height: 1.8;
          list-style-type: decimal;
        }

        .cross-validation ol li {
          margin-bottom: 1rem;
        }

        </style>
        <div class="container">
          <div class="section">
            <div class="header">
              <h1>Model Training, Evaluation, and Comparison</h1>
            </div>

            <div class="model-cards">
              <div class="model-card">
                <h2>Logistic Regression (LR)</h2>
                <p><strong>Strengths:</strong> Simple to interpret, works well with linearly separable data.</p>
                <p><strong>Possible Performance:</strong> The reduced features (AE-1, AE-2, AE-3, AE-4) can be linearly separated between status 0 and 1 according to the pairplot, and LR could achieve good accuracy.</p>
              </div>

              <div class="model-card">
                <h2>Linear Discriminant Analysis (LDA)</h2>
                <p><strong>Strengths:</strong> Assumes Gaussian distribution for features within each class. Often performs well for data with well-defined class clusters.</p>
                <p><strong>Possible Performance:</strong> Given our findings of (potentially) normal features and low correlation, LDA could be a strong contender.</p>
              </div>

              <div class="model-card">
                <h2>K-Nearest Neighbors (KNN)</h2>
                <p><strong>Strengths:</strong> No explicit assumption about data distribution. Effective for complex decision boundaries.</p>
                <p><strong>Possible Performance:</strong> KNN might be a good option if the class separation is not easily captured by a linear model. KNN can be sensitive to the choice of the "k" parameter.</p>
              </div>

              <div class="model-card">
                <h2>Gaussian Naive Bayes (GNB)</h2>
                <p><strong>Strengths:</strong> Fast and efficient, works well with conditional independence assumptions between features.</p>
                <p><strong>Possible Performance:</strong> GNB assumes features are independent given the class label, which might not be the case based on our observation of low correlation.</p>
              </div>
            </div>

            <div class="overall">
              <p><strong>Overall:</strong> Given the findings from our EDA, LDA seems like a promising choice due to its assumptions aligning well with the data characteristics.</p>
            </div>
          </div>

          <div class="section">
            <h2 class="heading">Cross Validation</h2>

            <div class="cross-validation">
              <ol>
                <li><strong>Data Splitting:</strong> Divide the train set into 5 equal folds.</li>
                <li><strong>K-Fold Loop (K=5):</strong> Iterate through each fold, combining 4 folds for training and using the remaining fold as validation.</li>
                <li><strong>Hyperparameter Tuning:</strong> Define a grid of hyperparameters and test each combination within each fold.</li>
                <li><strong>Model Selection:</strong> Select the model with the lowest maximum validation MSE across all folds.</li>
              </ol>
            </div>
          </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def __(train_test_split, transformed_synthetic_data):
    # generate train & test data
    train, test = train_test_split(
        transformed_synthetic_data, test_size=0.2, random_state=42
    )
    return test, train


@app.cell(hide_code=True)
def __(cartesian_product_dicts):
    # params for cross-validation

    lr_params = cartesian_product_dicts(
        [
            [{"C": 10 ** (i - 4)} for i in range(7)],
            [
                {"solver": solver}
                for solver in [
                    "lbfgs",
                    "liblinear",
                    "newton-cg",
                    "newton-cholesky",
                    "sag",
                    "saga",
                ]
            ],
            [{"max_iter": 10_000}],
        ]
    )

    lda_params = [{"solver": solver} for solver in ["svd", "lsqr", "eigen"]]

    knn_params = cartesian_product_dicts(
        [
            [{"n_neighbors": n_neighbors} for n_neighbors in range(1, 10)],
            [{"weights": weights} for weights in ["uniform", "distance"]],
            [
                {"metric": metric}
                for metric in ["minkowski", "manhattan", "euclidean"]
            ],
        ]
    )

    gn_params = [
        {"var_smoothing": 10 ** (var_smoothing - 10)}
        for var_smoothing in range(10)
    ]
    return gn_params, knn_params, lda_params, lr_params


@app.cell(hide_code=True)
def __(KFold, np, product):
    # functions for cross-validation


    def cartesian_product_dicts(
        lists_of_dicts: list[list[dict[str, any]]],
    ) -> list[dict[str, any]]:
        """
        Compute the Cartesian product of lists of dictionaries.

        Args:
        - lists_of_dicts: A list of lists, each containing dictionaries.

        Returns:
        - A list of dictionaries, representing the Cartesian product.
        """
        # Compute Cartesian product
        cartesian_product = product(*lists_of_dicts)

        # Merge dictionaries in the product
        result = []
        for item in cartesian_product:
            merged_dict = {}
            for dictionary in item:
                merged_dict.update(dictionary)
            result.append(merged_dict)

        return result


    def cross_validation(data, n_fold, clf, params):
        kf = KFold(n_fold)
        result = np.zeros((n_fold, len(params)))
        for i, param in enumerate(params):
            for j, (train, valid) in enumerate(kf.split(data)):
                X_train, X_valid, y_train, y_valid = (
                    data[train][:, :-1],
                    data[valid][:, :-1],
                    data[train][:, -1],
                    data[valid][:, -1],
                )
                model = clf(**param)
                model.fit(X_train, y_train)
                result[j, i] = model.score(X_valid, y_valid)
        best_params = params[np.argmax(np.min(result, axis=0))]
        best_valid_score = max(np.min(result, axis=0))
        best_model = clf(**best_params).fit(data[:, :-1], data[:, -1])
        return {
            "param": best_params,
            "valid score": best_valid_score,
            "model": best_model,
        }
    return cartesian_product_dicts, cross_validation


@app.cell(hide_code=True)
def __(
    GaussianNB,
    KNeighborsClassifier,
    LinearDiscriminantAnalysis,
    LogisticRegression,
    cross_validation,
    gn_params,
    knn_params,
    lda_params,
    lr_params,
    train,
):
    # get cross-validation result
    lr_result = cross_validation(train, 5, LogisticRegression, lr_params)
    lda_result = cross_validation(train, 5, LinearDiscriminantAnalysis, lda_params)
    knn_result = cross_validation(train, 5, KNeighborsClassifier, knn_params)
    gn_result = cross_validation(train, 5, GaussianNB, gn_params)
    return gn_result, knn_result, lda_result, lr_result


@app.cell(hide_code=True)
def __(gn_result, knn_result, lda_result, lr_result):
    # prepare for summary
    models = [
        result["model"]
        for result in [lr_result, lda_result, knn_result, gn_result]
    ]
    valid_scores = [
        result["valid score"]
        for result in [lr_result, lda_result, knn_result, gn_result]
    ]
    return models, valid_scores


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The summary is as below:""")
    return


@app.cell(hide_code=True)
def __(models, summary_plot, test, valid_scores):
    summary_plot(models, valid_scores, test)
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo

    mo.md(
        r"""
        <h1 id="src">Source Code</h1>
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def __():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, Dataset
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torch.optim as optim
    import plotly.graph_objects as go
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import statistics
    return (
        DataLoader,
        Dataset,
        F,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        go,
        nn,
        np,
        optim,
        pd,
        random,
        statistics,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def __():
    from sklearn.linear_model import Lasso
    return (Lasso,)


@app.cell(hide_code=True)
def __():
    import plotly.figure_factory as ff
    import plotly.express as px
    return ff, px


@app.cell(hide_code=True)
def __(mo):
    import plotly.io as pio

    pio.templates.default = (
        "plotly_dark" if mo.app_meta().theme == "dark" else "simple_white"
    )
    return (pio,)


@app.cell(hide_code=True)
def __():
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import KFold
    return (
        GaussianNB,
        KFold,
        KNeighborsClassifier,
        LinearDiscriminantAnalysis,
        LogisticRegression,
    )


@app.cell(hide_code=True)
def __():
    from itertools import product
    return (product,)


@app.cell(hide_code=True)
def __():
    import polars as pl
    return (pl,)


@app.cell(hide_code=True)
def __():
    import anywidget
    import traitlets


    class HeaderWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
            const result = model.get("result");

            const container = document.createElement("div");
            container.className = "header-container";

            const banner = document.createElement("img");
            banner.className = "banner";
            banner.src = "https://raw.githubusercontent.com/Haleshot/marimo-tutorials/main/community-tutorials-banner.png";
            banner.style.width = "100%";
            banner.style.height = "200px";
            banner.style.objectFit = "cover";
            banner.style.borderRadius = "10px 10px 0 0";
            banner.alt = "Marimo Banner";

            const form = document.createElement("div");
            form.className = "form-container";

            for (const [key, value] of Object.entries(result)) {
                const row = document.createElement("div");
                row.className = "form-row";

                const label = document.createElement("label");
                label.textContent = key;

                const valueContainer = document.createElement("div");
                valueContainer.className = "value-container";

                if (value.length > 100) {
                    const preview = document.createElement("div");
                    preview.className = "preview";
                    preview.textContent = value.substring(0, 100) + "...";

                    const fullText = document.createElement("div");
                    fullText.className = "full-text";
                    fullText.textContent = value;

                    const toggleButton = document.createElement("button");
                    toggleButton.className = "toggle-button";
                    toggleButton.textContent = "Show More";
                    toggleButton.onclick = () => {
                        if (fullText.style.display === "none") {
                            fullText.style.display = "block";
                            preview.style.display = "none";
                            toggleButton.textContent = "Show Less";
                        } else {
                            fullText.style.display = "none";
                            preview.style.display = "block";
                            toggleButton.textContent = "Show More";
                        }
                    };

                    valueContainer.appendChild(preview);
                    valueContainer.appendChild(fullText);
                    valueContainer.appendChild(toggleButton);

                    fullText.style.display = "none";
                } else {
                    valueContainer.textContent = value;
                }

                row.appendChild(label);
                row.appendChild(valueContainer);
                form.appendChild(row);
            }

            container.appendChild(banner);
            container.appendChild(form);
            el.appendChild(container);
        }
        export default { render };
        """

        _css = """
        .header-container {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            max-width: 100%;
            margin: 0 auto;
            overflow: hidden;
        }

        .banner {
            width: 100%;
            height: auto;
            display: block;
        }

        .form-container {
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            font-weight: 300;
            box-shadow: 0 -10px 20px rgba(0,0,0,0.1);
        }

        .form-row {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
            font-weight: 500;
        }

        .value-container {
            font-size: 1em;
            line-height: 1.5;
        }

        .preview, .full-text {
            margin-bottom: 10px;
        }

        .toggle-button {
            border: none;
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .toggle-button:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        @media (max-width: 600px) {
            .form-container {
                grid-template-columns: 1fr;
            }
        }
        """

        result = traitlets.Dict({}).tag(sync=True)
    return HeaderWidget, anywidget, traitlets


@app.cell(hide_code=True)
def __(HeaderWidget):
    header_widget = HeaderWidget(
        result={
            "Title": "Exploring Parkinson's Disease Diagnosis",
            "Author": "H. Eugene",
            "Contact": "eugeneheiner14@gmail.com",
            "Date": "2024-09-25",
            "Keywords": "Synthetic Data Generation, Variational AutoEncoder, Dimensionality Reduction, Machine Learningg",
            "Data Sources": "https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set",
            "Tools Used": "Polars, PyTorch, Plotly",
            "Version": "0.1",
        }
    )
    return (header_widget,)


@app.cell(hide_code=True)
def __():
    # plotting Functions
    return


@app.cell(hide_code=True)
def __(go, pd, px):
    def correlation_heatmap(data: pd.DataFrame) -> go.Figure:
        return px.imshow(
            data.corr().round(2),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="teal",
        ).update_layout(title="Correlation Heatmap of Features")
    return (correlation_heatmap,)


@app.cell(hide_code=True)
def __(Lasso, go, np):
    def plot_lasso_coefficients(
        X: np.ndarray,
        y: np.ndarray,
        alpha_range: list[float],
        feature_names: list[str] = None,
    ) -> go.Figure:
        """
        Plots the coefficients of a Lasso model for different alpha values.

        Args:
            X: A numpy array of shape (n_samples, n_features) containing the features.
            y: A numpy array of shape (n_samples,) containing the target variable.
            alpha_range: A list of alpha values to use for fitting the Lasso model.
            feature_names: A list of strings of length n_features containing the names of the features. (Optional)

        Returns:
            go.Figure
        """

        models = []
        coefs = []
        for alpha in alpha_range:
            model = Lasso(alpha).fit(X, y)
            models.append(model)
            coefs.append(model.coef_)

            if feature_names is None:
                feature_names = [f"Feature {i+1}" for i in range(X.shape[1] - 1)]

        n_features = X.shape[1] - 1  # Exclude target column

        fig = go.Figure()
        for i in range(n_features):
            fig.add_trace(
                go.Scatter(
                    x=alpha_range,
                    y=np.array([coef[i] for coef in coefs]),
                    mode="lines",
                    name=feature_names[i],
                    showlegend=False,
                )
            )

        # Add threshold line (assuming zero is a reasonable threshold)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

        fig.update_traces(line_width=1)

        fig.update_layout(
            title="Lasso Coefficients vs. Alpha",
            xaxis={"title": "Alpha", "type": "log"},
            yaxis_title="Coefficient",
        )

        return fig
    return (plot_lasso_coefficients,)


@app.cell(hide_code=True)
def __(pd, px):
    def create_pairplot(data: pd.DataFrame, target="status"):
        data_copy = data.copy()
        data_copy[target] = data_copy[target].astype(str)
        return px.scatter_matrix(
            data_copy,
            dimensions=list(data_copy.columns[:-1]),
            color=target,
            symbol=target,
            symbol_sequence=["cross", "square"],
            opacity=0.7,
            title="Pairplot",
            color_discrete_sequence=["deeppink", "plum"],
        ).update_traces(
            marker_size=1,
        )
    return (create_pairplot,)


@app.cell(hide_code=True)
def __(go, np):
    def create_violin(data: np.ndarray):
        X, y = data[:, :-1], data[:, -1]
        labels = np.unique(y)
        names = [f"Class {label:.0f}" for label in labels]
        xpos = [f"AE-{_}" for _ in range(X.shape[1])]
        showlegend = [False for _ in range(X.shape[1] - 1)]
        showlegend.insert(0, True)
        fig = go.Figure()
        for col in range(X.shape[1]):
            for label, name, side, line_color, pointpos in zip(
                labels,
                names,
                ["negative", "positive"],
                ["deeppink", "plum"],
                [-1, 1],
            ):
                values = X[y == label][:, col]
                fig.add_trace(
                    go.Violin(
                        x=[xpos[col]] * len(values),
                        y=values,
                        name=name,
                        side=side,
                        line_color=line_color,
                        showlegend=showlegend[col],
                        pointpos=pointpos,
                    )
                )

        fig.update_traces(
            meanline_visible=True,
            points="all",  # show all points
            marker_size=1.4,
            jitter=0.3,  # add some jitter on points for better visibility
            scalemode="count",  # scale violin plot area with total count
        )  # scale violin plot area with total count

        fig.update_layout(
            title_text="Violin Plots",
            violingap=0,
            violingroupgap=0.33,
            violinmode="overlay",
        )
        return fig
    return (create_violin,)


@app.cell(hide_code=True)
def __(ff, np):
    def create_distplot(data: np.ndarray):
        X, y = data[:, :-1], data[:, -1]
        labels = np.unique(y)
        fig = ff.create_distplot(
            hist_data=[
                X[y == label][:, col]
                for label in labels
                for col in range(X.shape[1])
            ],
            group_labels=[
                f"class{label: .0f}-AE{_+1}"
                for label in labels
                for _ in range(X.shape[1])
            ],
            show_hist=False,
            show_rug=True,
        ).update_layout(
            title="Overlaying density",
        )
        return fig
    return (create_distplot,)


@app.cell(hide_code=True)
def __(px):
    def plot_train_loss(train_loss: list[float]) -> None:
        """
        Args:
            train_loss (list): List of training loss values.
        """
        return (
            px.line(
                x=list(range(1, len(train_loss) + 1)),
                y=train_loss,
            )
            .update_traces(
                line_width=1,
            )
            .update_layout(
                title="Training Loss Over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Loss",
            )
        )
    return (plot_train_loss,)


@app.cell(hide_code=True)
def __(pl, px):
    def summary_plot(models, valid_scores, test):
        # Prepare test set
        scores = valid_scores.copy()
        X, y = test[:, :-1], test[:, -1]

        # Append test scores for each model
        for model in models:
            scores.append(model.score(X, y))

        # Duplicate model names for both validation and test
        names = [str(model).split("(")[0] for model in models] * 2

        # Prepare labels for validation and test
        split = ["valid"] * len(valid_scores) + ["test"] * len(valid_scores)

        # Create DataFrame for scores, splits, and model names
        df = pl.DataFrame({"scores": scores, "split": split, "names": names})

        # Create a grouped dumbbell plot
        fig = px.scatter(
            df,  ## plotly expects a pandas DataFrame
            y="names",  ## group by model names
            x="scores",  ## x-axis represents the scores
            color="split",  ## color represents valid/test split
            symbol="split",  ## differentiate valid and test by symbol
            title="Validation vs Test Scores per Model",
        )

        # Add lines to connect validation and test scores for each model (dumbbell effect)

        for model in models:
            valid_score = valid_scores[models.index(model)]
            test_score = model.score(X, y)
            fig.add_shape(
                type="line",
                x0=valid_score,
                x1=test_score,
                y0=str(model).split("(")[0],
                y1=str(model).split("(")[0],
                line=dict(
                    color="gray", width=1, dash="dot"
                ),  ## aesthetic line style
            )

        # Update aesthetics (size, marker, etc.)
        fig.update_traces(marker_size=7)
        fig.update_layout(
            yaxis_title="Models", xaxis_title="Scores", showlegend=True
        )

        return fig
    return (summary_plot,)


@app.cell(hide_code=True)
def __():
    DATA_PATH = "assets/parkinsons.csv"
    return (DATA_PATH,)


if __name__ == "__main__":
    app.run()
