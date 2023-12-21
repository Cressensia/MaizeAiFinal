import {CognitoUserPool} from 'amazon-cognito-identity-js';

const poolData = {
    UserPoolId: 'ap-southeast-1_ORwzbHxDg', // Your user pool id here
    ClientId: "3mlrr5116s0sv474g7fapih9rd" // Your client id here
}

export default new CognitoUserPool(poolData);