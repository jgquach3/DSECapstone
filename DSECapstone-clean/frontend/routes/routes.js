import { Home, ContentPaste, Notifications, AccountCircle } from '@material-ui/icons';
import HomePage from '../pages/Home/HomePage';
import ProfilePage from '../pages/Profile/ProfilePage';

const Routes = [
  {
    path: '/dashboard/home',
    sidebarName: 'Home',
    navbarName: 'Home',
    icon: Home,
    component: HomePage
  },
  {
    path: '/dashboard/profile',
    sidebarName: 'Profile',
    navbarName: 'Profile',
    icon: AccountCircle,
    component: ProfilePage
  }
];

export default Routes;