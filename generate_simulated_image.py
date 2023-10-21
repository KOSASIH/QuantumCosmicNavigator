import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the simulated image
object_type = 'star'
position = (100, 100)
observation_conditions = {
    'exposure_time': 10,  # in seconds
    'telescope_diameter': 2.5,  # in meters
    'atmospheric_turbulence': 0.5  # on a scale of 0 to 1
}

# Define a function to generate the simulated image
def generate_simulated_image(object_type, position, observation_conditions):
    # Use deep learning models or image synthesis techniques to generate the image
    # Here, we'll use a simple approach of generating a Gaussian point spread function (PSF)
    # centered at the specified position
    
    # Generate a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(200), np.arange(200))
    
    # Calculate the distance from each pixel to the specified position
    distance = np.sqrt((x - position[0])**2 + (y - position[1])**2)
    
    # Generate the PSF based on the observation conditions
    psf = np.exp(-0.5 * (distance / (observation_conditions['telescope_diameter'] / 2.355))**2)
    
    # Scale the PSF based on the exposure time
    psf *= observation_conditions['exposure_time']
    
    # Add atmospheric turbulence effects
    psf += np.random.normal(0, observation_conditions['atmospheric_turbulence'], psf.shape)
    
    # Normalize the image
    psf /= np.max(psf)
    
    return psf

# Generate the simulated image
simulated_image = generate_simulated_image(object_type, position, observation_conditions)

# Display the simulated image
plt.imshow(simulated_image, cmap='gray')
plt.axis('off')
plt.show()
