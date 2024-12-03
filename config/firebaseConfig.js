import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, onAuthStateChanged } from "firebase/auth";

const firebaseConfig = {
    apiKey: "AIzaSyA0jZcDLS9b31fVf5Wb3yk4XNBaC9lVYkw",       
    authDomain: "eduai-17e96.firebaseapp.com",        
    projectId: "eduai-17e96",        
    storageBucket: "eduai-17e96.firebasestorage.app",     
    messagingSenderId: "46846141128",    
    appId: "1:46846141128:web:9671893770e1d546a33bfd",  
    measurementId: "G-993XBM7FB2"
  };

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// User sign-in example
async function signIn(email, password) {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    console.log('Signed in as:', userCredential.user);
  } catch (error) {
    console.error('Error signing in:', error.message);
  }
}

// User registration example
async function signUp(email, password) {
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    console.log('User created:', userCredential.user);
  } catch (error) {
    console.error('Error signing up:', error.message);
  }
}

// Listen for authentication state changes
onAuthStateChanged(auth, (user) => {
  if (user) {
    console.log('User logged in:', user);
  } else {
    console.log('No user logged in');
  }
});
