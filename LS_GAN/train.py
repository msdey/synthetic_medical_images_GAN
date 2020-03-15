from preprocessing import *
from model import *

import time

mean_disc_loss = deque([0], maxlen=250)     
avg_gan_loss = deque([0], maxlen=250) 
X_train = get_images(image_path = x_ray_data)
train_size = X_train.shape[0]


discriminator_weights = './saved_weights/discriminator_model_lsgan.h5'

if os.path.exists(discriminator_weights):
  discriminator.load_weights(discriminator_weights)
  
for step in range(num_steps): 
    tot_step = step
    if step%500==0:
      print("Begin step: ", tot_step)
      begin_time = time.time() 

    # generate real and fake images
    random_index = np.random.randint(0, train_size , size = batch_size)
    real_data_X = X_train[random_index]
    noise = generate_noise(batch_size,noise_shape)
    fake_data_X = generator.predict(noise)

    #concatenate real and fake data samples
    X = np.concatenate([real_data_X,fake_data_X])
    

    real_data_Y = np.ones([batch_size, 1])
    fake_data_Y = np.zeros([batch_size, 1])

    Y = np.concatenate([real_data_Y, fake_data_Y])


    #Train the discriminator first
    discriminator.trainable = True
    gan.trainable = False
    disc_loss_real = discriminator.train_on_batch(real_data_X, real_data_Y)
    disc_loss_fake = discriminator.train_on_batch(fake_data_X, fake_data_Y)
    #disc_loss = discriminator.train_on_batch(X, Y)
    disc_loss = 0.5*np.add(disc_loss_real, disc_loss_fake)

    mean_disc_loss.append(disc_loss[0])

    if (tot_step % 500) == 0:
        step_num = str(tot_step).zfill(4)
        save_name = os.path.join(img_save_dir + '/' +  str(step_num)+  '.png')
        #save_img_batch(fake_data_X,img_save_dir+step_num+"_image.png")
        save_img_batch(fake_data_X, save_name)

    # generator trainable
    discriminator.trainable = False
    gan.trainable = True
    Y = np.ones([batch_size, 1])
    generated_noise = generate_noise(batch_size, noise_shape)
    gan_loss = gan.train_on_batch(generated_noise, Y) 

    text_file = open(log_dir+"\\training_log_lsgan.txt", "a")
    text_file.write("Step: %d Disc loss: %f GAN loss: %f\n" % (tot_step, disc_loss[0], gan_loss[0]))
    text_file.close()
    
    avg_gan_loss.append(gan_loss[0])

    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Discriminator loss: %f" % (np.mean(mean_disc_loss)))       
        print("Average GAN loss: %f" % (np.mean(avg_gan_loss)))
        print("-----------------------------------------------------------------")
        discriminator.save_weights(discriminator_weights)

